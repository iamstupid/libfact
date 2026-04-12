#!/usr/bin/env python3
"""Generate balanced semiprimes for benchmarking ECM stage 2."""
import sympy

# For each total bit size, generate p*q where p,q are each half-bit primes.
for total_bits in range(60, 201, 20):
    half = total_bits // 2
    p = sympy.nextprime(2**(half-1) + 1000)
    q = sympy.nextprime(p + 10000)
    n = p * q
    # Output as hex limbs (little-endian uint64)
    limbs = []
    v = n
    while v > 0:
        limbs.append(v & ((1 << 64) - 1))
        v >>= 64
    N = max((total_bits + 127) // 128, 1)  # UInt<N> needed
    if total_bits <= 64: N = 1
    elif total_bits <= 128: N = 2
    elif total_bits <= 192: N = 3
    else: N = 4
    while len(limbs) < N:
        limbs.append(0)
    limb_str = ", ".join(f"0x{l:016X}ULL" for l in limbs[:N])
    print(f"    // {total_bits}bit = {p} * {q}")
    print(f"    // p={p} ({p.bit_length()}bit), q={q} ({q.bit_length()}bit)")
    print(f"    {{ {total_bits}, {N}, {{ {limb_str} }} }},")
