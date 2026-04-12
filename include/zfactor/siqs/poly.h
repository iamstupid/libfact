#pragma once

// SIQS Polynomial Generation with Contini Self-Initialization.
//
// For Q(x) = A*x^2 + 2*B*x + C  where C = (B^2 - kn) / A (exact).
// We need B^2 ≡ kn (mod A), so A = product of s factor base primes q_j,
// and B is constructed via CRT.
//
// For each A, we generate 2^(s-1) distinct B values using Gray code
// switching: each switch flips one B_l value (add or subtract), and only
// requires one modular add/sub per factor base root to update sieve positions.
//
// The polynomial value Q(x) = (A*x + B)^2 - kn  (divided by A gives
// a*x^2 + 2*b*x + c which is what we actually sieve).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/gcd.h"
#include "zfactor/fixint/iroot.h"
#include "zfactor/siqs/factor_base.h"
#include "zfactor/siqs/params.h"

namespace zfactor::siqs {

// Maximum number of A-factors
static constexpr int MAX_A_FACTORS = 20;

template<int N>
struct SiqsPoly {
    fixint::UInt<N> A;
    fixint::UInt<N> B;
    // C = (B^2 - kn) / A stored as signed value
    fixint::UInt<N> C_abs;
    bool C_neg;

    // A-factor bookkeeping
    uint32_t a_factors[MAX_A_FACTORS];  // FB indices of primes in A
    int s;                               // number of A-factors

    // Gray code intermediates B_l[0..s-1]
    fixint::UInt<N> Bl[MAX_A_FACTORS];

    // Per-FB-prime precomputed values for root switching:
    //   ainv2[i] = (2*A)^(-1) mod prime[i]
    //   b_over_a[l][i] = Bl[l] * ainv2[i] mod prime[i]  (for Gray code switch)
    std::vector<uint32_t> ainv2;          // length = fb_size
    std::vector<std::vector<uint32_t>> b_over_a;  // [s][fb_size]

    // Current polynomial roots: root1[i], root2[i] for FB prime i
    // These are sieve offsets: location x in [-M, M] such that prime[i] | Q(x).
    std::vector<int32_t> root1, root2;

    // Current B-poly index and total
    int poly_index;
    int num_b_polys;

    // Gray code tables
    std::vector<int> gray_nu;     // which Bl to toggle
    std::vector<int> gray_sign;   // +1 or -1
};

// modinv for 32-bit values (wrapper around existing modinv_u64)
inline uint32_t modinv_u32(uint32_t a, uint32_t p) {
    return (uint32_t)fixint::modinv_u64((uint64_t)a, (uint64_t)p);
}

// modadd/modsub helpers
inline uint32_t modadd(uint32_t a, uint32_t b, uint32_t p) {
    uint64_t s = (uint64_t)a + b;
    return s >= p ? (uint32_t)(s - p) : (uint32_t)s;
}
inline uint32_t modsub(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? a - b : (uint32_t)((uint64_t)a + p - b);
}

// Build Gray code tables for 2^(s-1) polynomials.
inline void build_gray_code(int s, std::vector<int>& nu, std::vector<int>& sign) {
    int count = 1 << (s - 1);
    nu.resize(count);
    sign.resize(count);
    nu[0] = 0;
    sign[0] = 0;
    for (int i = 1; i < count; i++) {
        // Find position of lowest set bit in i
        int v = 0, j = i;
        while ((j & 1) == 0) { v++; j >>= 1; }
        nu[i] = v;
        // Determine sign: check bit at position v+1 in i
        int tmp = (i >> (v + 1));
        sign[i] = (tmp & 1) ? -1 : +1;
    }
}

// Choose a new A value from factor base primes.
// Strategy (from msieve): pick s-1 primes randomly from a pool of ~10-15 bit
// primes in the FB, then choose the last prime to bring A close to target_a.
template<int N>
void new_poly_a(SiqsPoly<N>& poly, const FactorBase& fb,
                const fixint::UInt<N>& kn, uint32_t sieve_size,
                uint32_t& seed1, uint32_t& seed2) {
    // Compute target_a = sqrt(2*kn) / sieve_size
    fixint::UInt<N> two_kn;
    {
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            unsigned __int128 w = (unsigned __int128)kn.d[i] * 2 + carry;
            two_kn.d[i] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
    }
    fixint::UInt<N> target_a = fixint::isqrt(two_kn);
    // Divide target_a by sieve_size (single-limb divide)
    {
        uint64_t rem = 0;
        for (int i = N - 1; i >= 0; i--) {
            unsigned __int128 w = ((unsigned __int128)rem << 64) | target_a.d[i];
            target_a.d[i] = (uint64_t)(w / sieve_size);
            rem = (uint64_t)(w % sieve_size);
        }
    }
    uint32_t target_bits = target_a.bit_length();

    // Determine number of A-factors and target bits per factor
    int s;
    int start_bits;
    if (target_bits > 210) { start_bits = 15; }
    else if (target_bits > 180) { start_bits = 12; }
    else if (target_bits > 140) { start_bits = 11; }
    else if (target_bits > 100) { start_bits = 10; }
    else { start_bits = 9; }

    s = std::max(3, (int)(target_bits / start_bits));
    if (s > MAX_A_FACTORS) s = MAX_A_FACTORS;
    poly.s = s;

    // Find the pool of FB primes with the right bit size.
    // Pool: primes between ~2000 and fb.med_B, with logp near start_bits.
    uint32_t pool_lo = 2, pool_hi = std::min(fb.med_B, fb.fb_size);
    // Narrow to primes of appropriate size
    for (uint32_t i = 2; i < fb.fb_size; i++) {
        if (fb.prime[i] >= 500) { pool_lo = i; break; }
    }
    for (uint32_t i = pool_lo; i < fb.fb_size; i++) {
        if (fb.prime[i] >= (1u << 16)) { pool_hi = i; break; }
    }
    if (pool_hi <= pool_lo + s + 2) {
        pool_lo = 2;
        pool_hi = std::min(fb.med_B, fb.fb_size);
    }

    // Marsaglia MWC random number generator
    auto get_rand = [&]() -> uint32_t {
        uint64_t temp = (uint64_t)seed1 * 2131995753ULL + seed2;
        seed1 = (uint32_t)temp;
        seed2 = (uint32_t)(temp >> 32);
        return seed1;
    };

    // Pick s-1 random distinct primes from the pool
    fixint::UInt<N> A;
    A.d[0] = 1;
    for (int i = 1; i < N; i++) A.d[i] = 0;

    bool chosen[MAX_A_FACTORS] = {};
    int filled = 0;

    auto try_pick = [&]() {
        filled = 0;
        A.d[0] = 1; for (int i = 1; i < N; i++) A.d[i] = 0;

        for (int attempt = 0; attempt < 100 && filled < s - 1; attempt++) {
            uint32_t idx = pool_lo + get_rand() % (pool_hi - pool_lo);
            // Check for duplicates
            bool dup = false;
            for (int j = 0; j < filled; j++)
                if (poly.a_factors[j] == idx) { dup = true; break; }
            if (dup) continue;

            poly.a_factors[filled] = idx;
            // Multiply A by this prime
            uint32_t p = fb.prime[idx];
            uint64_t carry = 0;
            for (int i = 0; i < N; i++) {
                unsigned __int128 w = (unsigned __int128)A.d[i] * p + carry;
                A.d[i] = (uint64_t)w;
                carry = (uint64_t)(w >> 64);
            }
            filled++;
        }
    };

    try_pick();
    if (filled < s - 1) {
        // Relax constraints
        pool_lo = 2;
        try_pick();
    }

    // Choose last factor to bring A close to target_a
    // Compute desired last factor = target_a / A
    fixint::UInt<N> desired_last;
    {
        fixint::UInt<N> q, r;
        fixint::mpn::divrem<N>(q.d, r.d, target_a.d, A.d);
        desired_last = q;
    }
    uint64_t desired = desired_last.d[0];

    // Find the FB prime closest to desired
    uint32_t best_idx = pool_lo;
    uint64_t best_dist = UINT64_MAX;
    for (uint32_t i = 2; i < pool_hi; i++) {
        // Skip if already chosen
        bool skip = false;
        for (int j = 0; j < filled; j++)
            if (poly.a_factors[j] == i) { skip = true; break; }
        if (skip) continue;

        uint64_t p = fb.prime[i];
        uint64_t dist = (p > desired) ? (p - desired) : (desired - p);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    poly.a_factors[filled] = best_idx;
    {
        uint32_t p = fb.prime[best_idx];
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            unsigned __int128 w = (unsigned __int128)A.d[i] * p + carry;
            A.d[i] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
    }
    filled++;
    poly.s = filled;

    // Sort factors
    std::sort(poly.a_factors, poly.a_factors + poly.s);
    poly.A = A;

    // Setup Gray code
    poly.num_b_polys = 1 << (poly.s - 1);
    poly.poly_index = 0;
    build_gray_code(poly.s, poly.gray_nu, poly.gray_sign);
}

// Initialize the first B-polynomial for a given A.
// Computes all B_l values and the initial sieve roots.
template<int N>
void init_poly_b(SiqsPoly<N>& poly, const FactorBase& fb,
                 const fixint::UInt<N>& kn, uint32_t sieve_len) {
    int s = poly.s;
    auto& A = poly.A;

    // For each A-factor q_j with FB index a_factors[j]:
    //   B_l[j] = (A/q_j) * modinv(A/q_j, q_j) * modsqrt(kn, q_j)
    // where each B_l[j] is a multi-precision number.

    for (int j = 0; j < s; j++) {
        uint32_t qidx = poly.a_factors[j];
        uint32_t q = fb.prime[qidx];
        uint32_t sq = fb.modsqrt[qidx];  // sqrt(kn) mod q

        // Compute A/q (exact division)
        fixint::UInt<N> A_div_q;
        {
            uint64_t rem = 0;
            for (int i = N - 1; i >= 0; i--) {
                unsigned __int128 w = ((unsigned __int128)rem << 64) | A.d[i];
                A_div_q.d[i] = (uint64_t)(w / q);
                rem = (uint64_t)(w % q);
            }
        }

        // inv = modinv(A_div_q mod q, q)
        uint32_t a_div_q_modq = uint_mod_u32<N>(A_div_q, q);
        uint32_t inv = modinv_u32(a_div_q_modq, q);

        // r = inv * sq mod q, normalized so r <= q/2
        uint32_t r = (uint32_t)(((uint64_t)inv * sq) % q);
        if (r > q / 2) r = q - r;

        // B_l[j] = A_div_q * r
        poly.Bl[j] = A_div_q;
        {
            uint64_t carry = 0;
            for (int i = 0; i < N; i++) {
                unsigned __int128 w = (unsigned __int128)poly.Bl[j].d[i] * r + carry;
                poly.Bl[j].d[i] = (uint64_t)w;
                carry = (uint64_t)(w >> 64);
            }
        }
    }

    // B = sum of all B_l[j]
    poly.B = {};
    for (int j = 0; j < s; j++) {
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            unsigned __int128 w = (unsigned __int128)poly.B.d[i] + poly.Bl[j].d[i] + carry;
            poly.B.d[i] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
    }

    // Double each B_l for the Gray code switching (each switch adds/subs 2*B_l)
    for (int j = 0; j < s; j++) {
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            unsigned __int128 w = (unsigned __int128)poly.Bl[j].d[i] * 2 + carry;
            poly.Bl[j].d[i] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
    }

    // Compute C = (B^2 - kn) / A
    {
        fixint::UInt<2*N> B2;
        fixint::mpn::mul<N>(B2.d, poly.B.d, poly.B.d);
        // Compare B^2 with kn (padded to 2N limbs)
        fixint::UInt<2*N> kn_wide = {};
        for (int i = 0; i < N; i++) kn_wide.d[i] = kn.d[i];

        fixint::UInt<2*N> diff;
        if (fixint::mpn::cmp<2*N>(B2.d, kn_wide.d) >= 0) {
            fixint::mpn::sub<2*N>(diff.d, B2.d, kn_wide.d);
            poly.C_neg = false;
        } else {
            fixint::mpn::sub<2*N>(diff.d, kn_wide.d, B2.d);
            poly.C_neg = true;
        }
        // Exact divide by A (into N limbs since result fits)
        fixint::UInt<2*N> A_wide = {};
        for (int i = 0; i < N; i++) A_wide.d[i] = A.d[i];
        fixint::UInt<2*N> Q_wide, R_wide;
        fixint::mpn::divrem<2*N>(Q_wide.d, R_wide.d, diff.d, A_wide.d);
        for (int i = 0; i < N; i++) poly.C_abs.d[i] = Q_wide.d[i];
    }

    // Precompute per-prime quantities for sieve root computation.
    uint32_t fb_size = fb.fb_size;
    poly.ainv2.resize(fb_size, 0);
    poly.b_over_a.resize(s);
    for (int j = 0; j < s; j++) poly.b_over_a[j].resize(fb_size, 0);
    poly.root1.resize(fb_size, 0);
    poly.root2.resize(fb_size, 0);

    for (uint32_t i = 2; i < fb_size; i++) {
        uint32_t p = fb.prime[i];
        if (p == 0) continue;

        // Check if this prime divides A (skip sieving for A-factors)
        bool divides_a = false;
        for (int j = 0; j < s; j++) {
            if (poly.a_factors[j] == i) { divides_a = true; break; }
        }

        if (divides_a) {
            poly.root1[i] = -1;  // sentinel: don't sieve
            poly.root2[i] = -1;
            poly.ainv2[i] = 0;
            continue;
        }

        uint32_t a_modp = uint_mod_u32<N>(A, p);

        // ainv = A^{-1} mod p  (for root computation)
        uint32_t ainv = modinv_u32(a_modp, p);

        // ainv2 = (2*A)^{-1} mod p  (for Gray code root updates)
        uint32_t two_a_modp = (uint32_t)(((uint64_t)a_modp * 2) % p);
        poly.ainv2[i] = modinv_u32(two_a_modp, p);

        // b_modp = B mod p
        uint32_t b_modp = uint_mod_u32<N>(poly.B, p);

        // Two roots of Q(x) = 0 (mod p):
        //   (Ax + B)^2 ≡ kn (mod p)  =>  x ≡ A^{-1} * (±sqrt(kn) - B)
        uint32_t sq = fb.modsqrt[i];
        // x-coordinate roots (polynomial variable):
        uint32_t xr1 = (uint32_t)(((uint64_t)ainv *
                        modsub(sq, b_modp, p)) % p);
        uint32_t xr2 = (uint32_t)(((uint64_t)ainv *
                        modsub(p - sq, b_modp, p)) % p);

        // Convert to sieve offsets: sieve_loc = x + M, where M = half_sieve
        // We need the sieve root in [0, p) such that locations
        // root, root+p, root+2p, ... are divisible by p.
        // sieve_root = (x_root + M) mod p
        // But M is set by caller. For now store x-roots and let the sieve
        // code handle the offset.  Actually, msieve stores the sieve offset
        // directly. Let's compute it here using sieve_len/2 as M.
        // This is set once per A (init_poly_b is called with the right sieve params).
        //
        // For Gray code switching to work, the updates delta must also be
        // in sieve offset space, which they are since adding/subtracting
        // the same delta to an offset just shifts the progression.
        uint32_t M_modp = (uint32_t)((sieve_len / 2) % p);
        uint32_t r1 = modadd(xr1, M_modp, p);
        uint32_t r2 = modadd(xr2, M_modp, p);

        poly.root1[i] = (int32_t)r1;
        poly.root2[i] = (int32_t)r2;

        // Precompute b_over_a[l][i] = 2*B_l[j] * ainv2 mod p  for Gray code switch
        for (int j = 0; j < s; j++) {
            uint32_t bl_modp = uint_mod_u32<N>(poly.Bl[j], p);
            poly.b_over_a[j][i] = (uint32_t)(((uint64_t)bl_modp * poly.ainv2[i]) % p);
        }
    }

    poly.poly_index = 0;
}

// Advance to the next B-polynomial via Gray code.
// Updates poly.B, poly.root1, poly.root2 in place.
// Returns false if all B-polys for this A have been exhausted.
template<int N>
bool next_poly_b(SiqsPoly<N>& poly, const FactorBase& fb,
                 const fixint::UInt<N>& kn) {
    poly.poly_index++;
    if (poly.poly_index >= poly.num_b_polys)
        return false;

    int idx = poly.poly_index;
    int l = poly.gray_nu[idx];      // which B_l to toggle
    int sign = poly.gray_sign[idx]; // +1 or -1

    // Update B: B_new = B_old +/- 2*B_l[l] (the Bl are already doubled)
    if (sign > 0) {
        // B += Bl[l]
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            unsigned __int128 w = (unsigned __int128)poly.B.d[i] + poly.Bl[l].d[i] + carry;
            poly.B.d[i] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
    } else {
        // B -= Bl[l]
        uint64_t borrow = 0;
        for (int i = 0; i < N; i++) {
            unsigned __int128 w = (unsigned __int128)poly.B.d[i] - poly.Bl[l].d[i] - borrow;
            poly.B.d[i] = (uint64_t)w;
            borrow = (w >> 127) ? 1 : 0;  // detect borrow
        }
    }

    // Recompute C = (B^2 - kn) / A
    {
        fixint::UInt<2*N> B2;
        fixint::mpn::mul<N>(B2.d, poly.B.d, poly.B.d);
        fixint::UInt<2*N> kn_wide = {};
        for (int i = 0; i < N; i++) kn_wide.d[i] = kn.d[i];
        fixint::UInt<2*N> diff;
        if (fixint::mpn::cmp<2*N>(B2.d, kn_wide.d) >= 0) {
            fixint::mpn::sub<2*N>(diff.d, B2.d, kn_wide.d);
            poly.C_neg = false;
        } else {
            fixint::mpn::sub<2*N>(diff.d, kn_wide.d, B2.d);
            poly.C_neg = true;
        }
        fixint::UInt<2*N> A_wide = {};
        for (int i = 0; i < N; i++) A_wide.d[i] = poly.A.d[i];
        fixint::UInt<2*N> Q_wide, R_wide;
        fixint::mpn::divrem<2*N>(Q_wide.d, R_wide.d, diff.d, A_wide.d);
        for (int i = 0; i < N; i++) poly.C_abs.d[i] = Q_wide.d[i];
    }

    // Update sieve roots: O(1) per prime.
    // root_new = root_old +/- b_over_a[l][i]  (mod prime[i])
    for (uint32_t i = 2; i < fb.fb_size; i++) {
        if (poly.root1[i] == -1) continue;  // A-factor, skip

        uint32_t p = fb.prime[i];
        uint32_t delta = poly.b_over_a[l][i];

        uint32_t r1 = (uint32_t)poly.root1[i];
        uint32_t r2 = (uint32_t)poly.root2[i];

        if (sign > 0) {
            r1 = modsub(r1, delta, p);
            r2 = modsub(r2, delta, p);
        } else {
            r1 = modadd(r1, delta, p);
            r2 = modadd(r2, delta, p);
        }

        poly.root1[i] = (int32_t)r1;
        poly.root2[i] = (int32_t)r2;
    }

    return true;
}

} // namespace zfactor::siqs
