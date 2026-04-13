#!/usr/bin/env python3
"""Generate LLVM IR for in-place divexact1 peel loop.

For each N, emits a function:
  i32 @divexact1_ip_peel_N(ptr %n, i64 %inv, i64 %d, ptr %tmp)

Semantics:
  Repeatedly divides n[] in-place by d (using modular inverse inv).
  After each successful division, copies n[] to tmp[].
  On failure (non-exact), restores n[] from tmp[] and returns peel count.

The loop body is: n[i] = (n[i] - borrow) * inv, borrow = hi(q * d) + sub_borrow.
LLVM's backend will use mulx + adcq for the carry chain.

Output: .ll file, then compile via:
  llc-19 -O2 -march=native -filetype=asm -o divexact_x86_64.S divexact_ir.ll
"""

import argparse
from pathlib import Path


def bits(n: int) -> int:
    return 64 * n


def emit_divexact_ip_body(n: int) -> str:
    """Emit the in-place divexact body: n /= d, return borrow as i64."""
    nb = bits(n)
    lines = []
    a = lines.append

    # Load n as wide int, load inv and d
    a(f"  ; Load n[0..{n-1}] as i{nb}")
    a(f"  %n_wide_ptr = bitcast ptr %n to ptr")
    a(f"  %n_wide = load i{nb}, ptr %n_wide_ptr, align 8")

    # We need to process limb-by-limb with borrow chain.
    # In LLVM IR, the cleanest way is to work with wide integers and let
    # LLVM's legalization split them into limbs with proper carry handling.
    #
    # divexact algorithm:
    #   borrow = 0
    #   for i in 0..N-1:
    #     s = n[i] - borrow
    #     sub_borrow = (n[i] < borrow) ? 1 : 0
    #     q = s * inv    (mod 2^64)
    #     n[i] = q
    #     borrow = mulhi(q, d) + sub_borrow
    #
    # In wide-int IR: this is essentially (n_wide * inv_wide) mod 2^{64N},
    # but with the borrow check requiring the full product.
    #
    # Actually it's simpler to emit per-limb IR and let LLVM handle it.

    # Per-limb approach with explicit i64 operations
    a(f"  %inv64 = add i64 %inv, 0")  # identity, just to have a clean name
    a(f"  %d64 = add i64 %d, 0")

    borrow = "0"  # SSA name for current borrow (i64)
    for i in range(n):
        pfx = f"l{i}"
        # Load n[i]
        a(f"  %{pfx}_nptr = getelementptr i64, ptr %n, i64 {i}")
        a(f"  %{pfx}_orig = load i64, ptr %{pfx}_nptr, align 8")

        # s = n[i] - borrow, sub_borrow = (n[i] < borrow)
        a(f"  %{pfx}_s = sub i64 %{pfx}_orig, %{borrow}")
        a(f"  %{pfx}_cf = icmp ult i64 %{pfx}_orig, %{borrow}")
        a(f"  %{pfx}_cfext = zext i1 %{pfx}_cf to i64")

        # q = s * inv (lo 64 bits)
        a(f"  %{pfx}_q = mul i64 %{pfx}_s, %inv64")

        # Store q to n[i]
        a(f"  store i64 %{pfx}_q, ptr %{pfx}_nptr, align 8")

        # borrow = mulhi(q, d) + sub_borrow
        # mulhi via zext to i128, mul, lshr
        a(f"  %{pfx}_q128 = zext i64 %{pfx}_q to i128")
        a(f"  %{pfx}_d128 = zext i64 %d64 to i128")
        a(f"  %{pfx}_prod = mul i128 %{pfx}_q128, %{pfx}_d128")
        a(f"  %{pfx}_hi128 = lshr i128 %{pfx}_prod, 64")
        a(f"  %{pfx}_hi = trunc i128 %{pfx}_hi128 to i64")
        a(f"  %{pfx}_borrow = add i64 %{pfx}_hi, %{pfx}_cfext")

        borrow = f"%{pfx}_borrow"

    a(f"  ret i64 {borrow}")
    return "\n".join(lines)


def emit_peel_loop(n: int) -> str:
    """Emit in-place peel loop with scalar save to tmp."""
    lines = []
    a = lines.append

    fname = f"divexact1_ip_peel_{n}"
    nbytes = n * 8

    a(f"; ============================================================")
    a(f"; {fname}: in-place peel loop for N={n}")
    a(f"; ============================================================")
    a(f"define i32 @{fname}(ptr noalias %n, i64 %inv, i64 %d, ptr noalias %tmp) #0 {{")
    a(f"entry:")
    a(f"  br label %loop")
    a(f"")
    a(f"loop:")
    a(f"  %e = phi i32 [ 0, %entry ], [ %e_next, %loop_continue ]")

    # In-place divexact body: read n[i], compute, write back to n[i]
    for i in range(n):
        pfx = f"l{i}"
        a(f"  %{pfx}_nptr = getelementptr i64, ptr %n, i64 {i}")
        a(f"  %{pfx}_orig = load i64, ptr %{pfx}_nptr, align 8")

        borrow_name = "0" if i == 0 else f"%l{i-1}_borrow"

        a(f"  %{pfx}_s = sub i64 %{pfx}_orig, {borrow_name}")
        a(f"  %{pfx}_cf = icmp ult i64 %{pfx}_orig, {borrow_name}")
        a(f"  %{pfx}_cfext = zext i1 %{pfx}_cf to i64")
        a(f"  %{pfx}_q = mul i64 %{pfx}_s, %inv")
        a(f"  store i64 %{pfx}_q, ptr %{pfx}_nptr, align 8")
        a(f"  %{pfx}_q128 = zext i64 %{pfx}_q to i128")
        a(f"  %{pfx}_d128 = zext i64 %d to i128")
        a(f"  %{pfx}_prod = mul i128 %{pfx}_q128, %{pfx}_d128")
        a(f"  %{pfx}_hi128 = lshr i128 %{pfx}_prod, 64")
        a(f"  %{pfx}_hi = trunc i128 %{pfx}_hi128 to i64")
        a(f"  %{pfx}_borrow = add i64 %{pfx}_hi, %{pfx}_cfext")

    last_borrow = f"%l{n-1}_borrow"
    a(f"")
    a(f"  %exact = icmp eq i64 {last_borrow}, 0")
    a(f"  br i1 %exact, label %loop_continue, label %loop_exit")
    a(f"")
    a(f"loop_continue:")
    a(f"  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp, ptr align 8 %n, i64 {nbytes}, i1 false)")
    a(f"  %e_next = add i32 %e, 1")
    a(f"  br label %loop")
    a(f"")
    a(f"loop_exit:")
    a(f"  %has_peeled = icmp ugt i32 %e, 0")
    a(f"  br i1 %has_peeled, label %do_restore, label %done")
    a(f"")
    a(f"do_restore:")
    a(f"  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %n, ptr align 8 %tmp, i64 {nbytes}, i1 false)")
    a(f"  br label %done")
    a(f"")
    a(f"done:")
    a(f"  ret i32 %e")
    a(f"}}")
    return "\n".join(lines)


PREAMBLE = """\
; Generated by tools/fixint/gen_divexact_ir.py — do not edit.
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

attributes #0 = { noinline nounwind }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-ll", required=True, help="Output .ll file")
    parser.add_argument("--output-s", help="Output .S file (runs llc)")
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=8)
    parser.add_argument("--llc", default="llc-19", help="Path to llc")
    parser.add_argument("--mcpu", default="x86-64-v3", help="Target CPU for llc")
    args = parser.parse_args()

    parts = [PREAMBLE]
    for n in range(args.min_n, args.max_n + 1):
        parts.append(emit_peel_loop(n))
        parts.append("")

    ll_path = Path(args.output_ll)
    ll_path.parent.mkdir(parents=True, exist_ok=True)
    ll_path.write_text("\n".join(parts))
    print(f"Generated {ll_path}")

    if args.output_s:
        import subprocess
        s_path = Path(args.output_s)
        cmd = [
            args.llc,
            "-O2",
            f"-mcpu={args.mcpu}",
            "-mattr=+bmi2",
            "-filetype=asm",
            "-o", str(s_path),
            str(ll_path),
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print(f"Generated {s_path}")


if __name__ == "__main__":
    main()
