#pragma once

// SIQS Square Root Phase.
//
// Given null-space dependencies from Block Lanczos, compute actual factors.
// For each dependency (subset of relations with exponent sum ≡ 0 mod 2):
//   X = product of (A_i * x_i + B_i) mod n
//   Y = product of p^(e_p/2) mod n
//   Then X² ≡ Y² (mod n), so gcd(X ± Y, n) may yield a factor.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/gcd.h"
#include "zfactor/siqs/relation.h"
#include "zfactor/siqs/factor_base.h"
#include "zfactor/siqs/params.h"

namespace zfactor::siqs {

// Plain modular multiplication for UInt<N>: (a * b) mod m.
// Uses schoolbook multiply into UInt<2N> then divrem.
template<int N>
fixint::UInt<N> modmul(const fixint::UInt<N>& a, const fixint::UInt<N>& b,
                       const fixint::UInt<N>& m) {
    fixint::UInt<2*N> prod;
    fixint::mpn::mul<N>(prod.d, a.d, b.d);
    fixint::UInt<2*N> m_wide = {};
    for (int i = 0; i < N; i++) m_wide.d[i] = m.d[i];
    fixint::UInt<2*N> q, r;
    fixint::mpn::divrem<2*N>(q.d, r.d, prod.d, m_wide.d);
    fixint::UInt<N> result;
    for (int i = 0; i < N; i++) result.d[i] = r.d[i];
    return result;
}

// Modular exponentiation: base^exp mod m
template<int N>
fixint::UInt<N> modpow(const fixint::UInt<N>& base, uint32_t exp,
                       const fixint::UInt<N>& m) {
    fixint::UInt<N> result = {};
    result.d[0] = 1;
    fixint::UInt<N> cur = base;
    // Reduce base mod m first
    if (fixint::mpn::cmp<N>(cur.d, m.d) >= 0) {
        fixint::UInt<N> q, r;
        fixint::mpn::divrem<N>(q.d, r.d, cur.d, m.d);
        cur = r;
    }
    uint32_t e = exp;
    while (e > 0) {
        if (e & 1) result = modmul<N>(result, cur, m);
        cur = modmul<N>(cur, cur, m);
        e >>= 1;
    }
    return result;
}

template<int N>
std::optional<fixint::UInt<N>> sqrt_phase(
    const fixint::UInt<N>& n,
    const fixint::UInt<N>& kn,
    uint32_t multiplier,
    const FactorBase& fb,
    const std::vector<Relation>& relations,
    const std::vector<fixint::UInt<N>>& poly_a_list,
    const std::vector<std::vector<fixint::UInt<N>>>& poly_b_list,
    const uint64_t* deps,
    uint32_t num_deps,
    uint32_t ncols,
    uint32_t half_sieve)
{
    using UintN = fixint::UInt<N>;

    for (uint32_t dep = 0; dep < num_deps; dep++) {
        uint64_t mask = (uint64_t)1 << dep;

        UintN X = {}; X.d[0] = 1;  // Start at 1
        std::vector<uint32_t> fb_counts(fb.fb_size, 0);
        std::vector<uint64_t> large_primes_list;

        for (uint32_t col = 0; col < ncols; col++) {
            if (!(deps[col] & mask)) continue;

            auto& rel = relations[col];

            auto& A = poly_a_list[rel.a_poly_idx];
            UintN B = {};
            if (rel.a_poly_idx < poly_b_list.size() &&
                rel.poly_idx < poly_b_list[rel.a_poly_idx].size()) {
                B = poly_b_list[rel.a_poly_idx][rel.poly_idx];
            }

            int32_t x = (int32_t)rel.sieve_offset - (int32_t)half_sieve;
            uint32_t abs_x = (uint32_t)std::abs(x);
            bool x_neg = (x < 0);

            UintN ax = {};
            {
                uint64_t carry = 0;
                for (int i = 0; i < N; i++) {
                    unsigned __int128 w = (unsigned __int128)A.d[i] * abs_x + carry;
                    ax.d[i] = (uint64_t)w;
                    carry = (uint64_t)(w >> 64);
                }
            }

            UintN val;
            if (!x_neg) {
                fixint::mpn::add<N>(val.d, ax.d, B.d);
            } else {
                if (fixint::mpn::cmp<N>(B.d, ax.d) >= 0) {
                    fixint::mpn::sub<N>(val.d, B.d, ax.d);
                } else {
                    fixint::mpn::sub<N>(val.d, ax.d, B.d);
                }
            }

            if (fixint::mpn::cmp<N>(val.d, n.d) >= 0) {
                UintN q, r;
                fixint::mpn::divrem<N>(q.d, r.d, val.d, n.d);
                val = r;
            }

            if (!val.is_zero())
                X = modmul<N>(X, val, n);

            // Debug: verify single relation for dep 0
            if (dep == 0 && col < 2) {
                // (Ax+B)^2 mod n
                UintN lhs = modmul<N>(val, val, n);
                // A * product(fb primes) * cofactors mod n
                UintN rhs = {}; rhs.d[0] = 1;
                for (auto fi : rel.fb_offsets) {
                    UintN p_val = {}; p_val.d[0] = fb.prime[fi];
                    if (p_val.d[0] == 0) { // sign row
                        // negate rhs: rhs = n - rhs
                        fixint::mpn::sub<N>(rhs.d, n.d, rhs.d);
                        continue;
                    }
                    rhs = modmul<N>(rhs, p_val, n);
                }
                // multiply by large primes
                for (int lpi = 0; lpi < 2; lpi++) {
                    if (rel.large_prime[lpi] > 1) {
                        UintN lp_val = {}; lp_val.d[0] = rel.large_prime[lpi];
                        rhs = modmul<N>(rhs, lp_val, n);
                    }
                }
                bool rel_ok = (fixint::mpn::cmp<N>(lhs.d, rhs.d) == 0);
                fprintf(stderr, "    verify rel %u: (Ax+B)^2=%lu, A*prod=%lu %s\n",
                        col, lhs.d[0], rhs.d[0], rel_ok ? "OK" : "FAIL");
            }

            for (auto fi : rel.fb_offsets)
                fb_counts[fi]++;

            for (int k = 0; k < 2; k++)
                if (rel.large_prime[k] > 1)
                    large_primes_list.push_back(rel.large_prime[k]);
        }

        // Compute Y = product of p^(count/2) mod n
        UintN Y = {}; Y.d[0] = 1;

        for (uint32_t fi = 0; fi < fb.fb_size; fi++) {
            uint32_t count = fb_counts[fi];
            if (count < 2) continue;
            uint32_t exp = count / 2;

            uint32_t p = fb.prime[fi];
            if (p == 0) continue;

            UintN base = {}; base.d[0] = p;
            UintN pw = modpow<N>(base, exp, n);
            Y = modmul<N>(Y, pw, n);
        }

        // Handle large primes
        {
            std::sort(large_primes_list.begin(), large_primes_list.end());
            for (size_t i = 0; i + 1 < large_primes_list.size(); i += 2) {
                if (large_primes_list[i] == large_primes_list[i+1]) {
                    UintN lp_val = {}; lp_val.d[0] = large_primes_list[i];
                    Y = modmul<N>(Y, lp_val, n);
                }
            }
        }

        // Try gcd(X - Y, n) and gcd(X + Y, n)
        auto try_gcd = [&](UintN& g) -> bool {
            if (g.is_zero()) return false;
            if (fixint::mpn::cmp<N>(g.d, n.d) == 0) return false;
            if (g.d[0] == 1 && g.bit_length() == 1) return false;

            // Remove multiplier
            if (multiplier > 1) {
                uint32_t g_mod_k = uint_mod_u32<N>(g, multiplier);
                uint64_t d = fixint::gcd_u64(multiplier, g_mod_k);
                if (d > 1) {
                    uint64_t rem = 0;
                    for (int i = N - 1; i >= 0; i--) {
                        unsigned __int128 w = ((unsigned __int128)rem << 64) | g.d[i];
                        g.d[i] = (uint64_t)(w / d);
                        rem = (uint64_t)(w % d);
                    }
                }
                if (g.d[0] == 1 && g.bit_length() == 1) return false;
            }
            return true;
        };

        UintN diff, sum;
        if (fixint::mpn::cmp<N>(X.d, Y.d) >= 0)
            fixint::mpn::sub<N>(diff.d, X.d, Y.d);
        else
            fixint::mpn::sub<N>(diff.d, Y.d, X.d);

        fixint::mpn::add<N>(sum.d, X.d, Y.d);
        if (fixint::mpn::cmp<N>(sum.d, n.d) >= 0)
            fixint::mpn::sub<N>(sum.d, sum.d, n.d);

        // Debug: verify X² ≡ Y² (mod n) for first dependency
        if (dep == 0) {
            int dep_count = 0;
            for (uint32_t col = 0; col < ncols; col++)
                if (deps[col] & mask) dep_count++;

            UintN X2 = modmul<N>(X, X, n);
            UintN Y2 = modmul<N>(Y, Y, n);
            bool squares_match = (fixint::mpn::cmp<N>(X2.d, Y2.d) == 0);
            fprintf(stderr, "  sqrt_phase dep 0: %d rels, X²==Y²? %s\n",
                    dep_count, squares_match ? "YES" : "NO");
            fprintf(stderr, "    X=%lu Y=%lu X²=%lu Y²=%lu\n",
                    X.d[0], Y.d[0], X2.d[0], Y2.d[0]);
        }

        auto g1 = fixint::gcd<N>(diff, n);
        if (try_gcd(g1)) return g1;

        auto g2 = fixint::gcd<N>(sum, n);
        if (try_gcd(g2)) return g2;
    }

    return std::nullopt;
}

} // namespace zfactor::siqs
