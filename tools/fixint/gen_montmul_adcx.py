#!/usr/bin/env python3
"""Generate hand-tuned CIOS montmul inline asm for N=3..6.

Algorithm: Montgomery CIOS with mulx + adcx/adox parallel carry chains
and lazy memory loads (a[], b[], mod[] never held in registers).

Register footprint: (N+2) accumulator + 3 ptrs + 3 scratch + rdx + r-out
                  = N+9 registers, fits the GPR file for N <= 6.

For N >= 5 we put `inv` in memory ("m" constraint) instead of a register
to free one slot.  For N == 6 we additionally drop the dedicated `zero`
register and replace `adcxq zero, t_top` / `adoxq zero, t_top+1` with
`setc/seto + add` (3 instructions instead of 1, but saves a register).
"""
import argparse
from pathlib import Path


def emit_zero_residual(t_cf: int, t_of: int, has_zero_reg: bool) -> list[str]:
    """Emit the final 'add CF/OF residuals into t_top' sequence.

    With a dedicated zero register: 2 instructions (adcx zero, t_cf; adox zero, t_of).
    Without it: setc/seto into a byte, movzx, add — 6 instructions total but
    they parallelise on Zen4.
    """
    if has_zero_reg:
        return [
            f'        "adcxq %[zero], %[t{t_cf}]\\n\\t"',
            f'        "adoxq %[zero], %[t{t_of}]\\n\\t"',
        ]
    # No zero reg: capture flags into byte regs (reusing lo/hi after they're dead),
    # then add as 64-bit values.  setc/seto don't touch CF/OF themselves so we can
    # do both before any add.  The first addq clobbers OF, so we must seto before
    # any addq.
    return [
        f'        "setc %b[lo]\\n\\t"',
        f'        "seto %b[hi]\\n\\t"',
        f'        "movzbq %b[lo], %[lo]\\n\\t"',
        f'        "movzbq %b[hi], %[hi]\\n\\t"',
        f'        "addq %[lo], %[t{t_cf}]\\n\\t"',
        f'        "addq %[hi], %[t{t_of}]\\n\\t"',
    ]


def emit_phase1(n: int, iter_idx: int, has_zero_reg: bool) -> list[str]:
    """Phase 1: t += a * b[iter_idx]
    For iter 0, t starts at 0 so we use single CF chain (no adoxq needed).
    For iter > 0, t already has values so we use parallel CF/OF chains.
    Final residual: CF → t[N], OF → t[N+1] (only iter > 0 needs the OF capture).
    """
    asm = []
    asm.append(f'        // Phase 1: t {"=" if iter_idx == 0 else "+="} a * b[{iter_idx}]')
    asm.append(f'        "movq {iter_idx*8}(%[b]), %%rdx\\n\\t"')
    if has_zero_reg:
        asm.append(f'        "xorl %k[zero], %k[zero]\\n\\t"')
    else:
        # Need to clear CF (and OF for iter > 0) before the chain.
        # xorl on a dead reg clears both CF and OF.
        asm.append(f'        "xorl %k[lo], %k[lo]\\n\\t"')
    if iter_idx == 0:
        # Single CF chain: t starts at 0, mulx writes directly
        asm.append(f'        "mulxq (%[a]), %[t0], %[t1]\\n\\t"')
        for j in range(1, n):
            top = j + 1
            asm.append(f'        "mulxq {j*8}(%[a]), %[lo], %[t{top}]\\n\\t"')
            asm.append(f'        "adcxq %[lo], %[t{j}]\\n\\t"')
        # Capture final CF and zero the next limb (no OF chain in iter 0)
        if has_zero_reg:
            asm.append(f'        "adcxq %[zero], %[t{n}]\\n\\t"')
        else:
            asm.append(f'        "setc %b[lo]\\n\\t"')
            asm.append(f'        "movzbq %b[lo], %[lo]\\n\\t"')
            asm.append(f'        "addq %[lo], %[t{n}]\\n\\t"')
        asm.append(f'        "movq $0, %[t{n+1}]\\n\\t"')
    else:
        # Parallel chains: each mulx feeds adcx (CF) and adox (OF) into adjacent limbs
        for j in range(n):
            asm.append(f'        "mulxq {j*8}(%[a]), %[lo], %[hi]\\n\\t"')
            asm.append(f'        "adcxq %[lo], %[t{j}]\\n\\t"')
            asm.append(f'        "adoxq %[hi], %[t{j+1}]\\n\\t"')
        # Final residuals
        asm += emit_zero_residual(n, n + 1, has_zero_reg)
    return asm


def emit_phase2() -> list[str]:
    return [
        '        // Phase 2: m = t[0] * neg_inv (rdx = m for next mulx chain)',
        '        "movq %[t0], %%rdx\\n\\t"',
        '        "imulq %[inv], %%rdx\\n\\t"',
    ]


def emit_phase3(n: int, has_zero_reg: bool) -> list[str]:
    """Phase 3: t += mod * m. t[0] becomes 0 by construction."""
    asm = []
    asm.append('        // Phase 3: t += mod * m  (t[0] -> 0)')
    if has_zero_reg:
        asm.append('        "xorl %k[zero], %k[zero]\\n\\t"')
    else:
        # Clear CF/OF using a dead reg (lo) — gets overwritten by next mulx
        asm.append('        "xorl %k[lo], %k[lo]\\n\\t"')
    for j in range(n):
        asm.append(f'        "mulxq {j*8}(%[mod]), %[lo], %[hi]\\n\\t"')
        asm.append(f'        "adcxq %[lo], %[t{j}]\\n\\t"')
        asm.append(f'        "adoxq %[hi], %[t{j+1}]\\n\\t"')
    asm += emit_zero_residual(n, n + 1, has_zero_reg)
    return asm


def emit_shift(n: int) -> list[str]:
    """Conceptual shift t >>= 64.  Implemented as mov chain.
    Zen4 mov-elim retires these in zero µops.
    """
    asm = ['        // Shift t down by 64 bits (mov-elim makes this free)']
    for j in range(n + 1):
        asm.append(f'        "movq %[t{j+1}], %[t{j}]\\n\\t"')
    asm.append(f'        "movq $0, %[t{n+1}]\\n\\t"')
    return asm


def emit_iteration(n: int, i: int, last: bool, has_zero_reg: bool) -> list[str]:
    asm = [f'        // ==================== Iteration {i} ====================']
    asm += emit_phase1(n, i, has_zero_reg)
    asm += emit_phase2()
    asm += emit_phase3(n, has_zero_reg)
    if not last:
        asm += emit_shift(n)
    return asm


def emit_cond_subtract_asm(n: int) -> str:
    """Conditional subtract as a SECOND inline asm block.

    Inputs:  t1..t_n  (the result candidate, in registers from prior block)
             t_{n+1}  (the high-overflow limb)
             mod      (pointer)
    Outputs: t1..t_n  modified in-place to the reduced value

    Save slots: we materialise N copies of t1..t_n via "+&r" so the
    sbbq chain can overwrite them, then cmovb from saved (in plain
    "r" registers) on borrow.
    """
    in_outs = ", ".join(f"&t{j+1}" for j in range(n))
    save_decls = ", ".join(f"o{j} = t{j+1}" for j in range(n))

    asm_lines = []
    # First sbbq must be subq
    asm_lines.append('        "subq (%[mod]), %[t1]\\n\\t"')
    for j in range(1, n):
        asm_lines.append(f'        "sbbq {j*8}(%[mod]), %[t{j+1}]\\n\\t"')
    asm_lines.append(f'        "sbbq $0, %[t{n+1}]\\n\\t"')
    for j in range(n):
        asm_lines.append(f'        "cmovbq %[o{j}], %[t{j+1}]\\n\\t"')

    in_constraints_t = ",\n          ".join(
        f'[t{j+1}] "+&r"(t{j+1})' for j in range(n)
    )
    out = []
    out.append('    // Conditional subtract done in a separate asm block — keeps')
    out.append('    // the body block within the 16-GPR budget for higher N.')
    out.append(f'    uint64_t {save_decls};')
    out.append('    asm(')
    out += asm_lines
    out.append(f'        : {in_constraints_t},')
    out.append(f'          [t{n+1}] "+&r"(t{n+1})')
    out.append('        : [mod] "r"(mod),')
    out.append('          ' + ", ".join(f'[o{j}] "r"(o{j})' for j in range(n)) + ',')
    out.append(f'          [m_mem] "m"(*(const uint64_t(*)[{n}])mod)')
    out.append('        : "cc"')
    out.append('    );')
    for j in range(n):
        out.append(f'    r[{j}] = t{j+1};')
    return '\n'.join(out)


def emit_function(n: int) -> str:
    # Register-budget switch:
    #   N <= 4 :  inv in reg, 3 scratch (lo, hi, zero)   → n+9 regs
    #   N == 5 :  inv in mem, 3 scratch                  → n+8 regs (fits 15 budget)
    #   N == 6 :  inv in mem, 2 scratch (no zero reg)    → n+7 regs (fits 15)
    #   N >= 7 :  not yet supported by this approach
    inv_in_mem = (n >= 5)
    has_zero_reg = (n <= 5)  # N>=6: drop zero reg via setc/seto residual capture
    inv_constraint = '"m"(neg_inv)' if inv_in_mem else '"r"(neg_inv)'
    scratch_decl = "uint64_t lo, hi, zero;" if has_zero_reg else "uint64_t lo, hi;"
    scratch_constraints = (
        '[lo]   "=&r"(lo),\n          [hi]   "=&r"(hi),\n          [zero] "=&r"(zero)'
        if has_zero_reg else
        '[lo]   "=&Q"(lo),\n          [hi]   "=&Q"(hi)'
    )

    body = []
    for i in range(n):
        body += emit_iteration(n, i, last=(i == n - 1), has_zero_reg=has_zero_reg)

    accumulator_decl = ", ".join(f"t{j} = 0" for j in range(n + 2))
    constraints_t = ",\n          ".join(f'[t{j}]   "+&r"(t{j})' for j in range(n + 2))

    footprint_note = (
        f"{n+2} t + 3 ptrs + 3 scratch + rdx + r-out = {n+9} regs (fits 16-GPR)"
        if not inv_in_mem else
        (f"{n+2} t + 3 ptrs + 2 scratch + rdx + r-out = {n+8} regs; "
         f"`inv` in mem, no `zero` reg (setc/seto residual capture)"
         if not has_zero_reg else
         f"{n+2} t + 3 ptrs + 3 scratch + rdx + r-out = {n+9} regs; "
         f"`inv` in mem")
    )

    return f"""\
// ============================================================================
// montmul_{n}_adcx — generated by tools/fixint/gen_montmul_adcx.py.
// Hand-tuned CIOS for N={n} using mulx + adcx/adox parallel carry chains.
// Lazy memory loads: a[], b[], mod[] are never held in registers.
// Footprint: {footprint_note}.
// ============================================================================
__attribute__((always_inline))
inline void montmul_{n}_adcx(mpn::limb_t* __restrict__ r,
                           const mpn::limb_t* __restrict__ a,
                           const mpn::limb_t* __restrict__ b,
                           const mpn::limb_t* __restrict__ mod,
                           mpn::limb_t neg_inv) {{
    uint64_t {accumulator_decl};
    {scratch_decl}

    asm(
{chr(10).join(body)}

        : {constraints_t},
          {scratch_constraints}
        : [a]      "r"(a),
          [b]      "r"(b),
          [mod]    "r"(mod),
          [inv]    {inv_constraint},
          [a_mem]  "m"(*(const uint64_t(*)[{n}])a),
          [b_mem]  "m"(*(const uint64_t(*)[{n}])b),
          [m_mem]  "m"(*(const uint64_t(*)[{n}])mod)
        : "rdx", "cc"
    );

{emit_cond_subtract_asm(n)}
}}
"""


def emit_header(min_n: int, max_n: int) -> str:
    out = [
        '// Generated by tools/fixint/gen_montmul_adcx.py — do not edit.',
        '// Hand-tuned CIOS Montgomery multiplication for N=3..{} using',
        '// mulx + adcx/adox parallel carry chains and lazy memory loads.',
        '#pragma once',
        '',
        '#include <cstdint>',
        '#include "../mpn.h"',
        '',
        'namespace zfactor::fixint {',
        '',
    ]
    for n in range(min_n, max_n + 1):
        out.append(emit_function(n))
        out.append('')
    out.append('} // namespace zfactor::fixint')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--min-n', type=int, default=3)
    ap.add_argument('--max-n', type=int, default=4)
    args = ap.parse_args()

    text = emit_header(args.min_n, args.max_n)
    Path(args.output).write_text(text)


if __name__ == '__main__':
    main()
