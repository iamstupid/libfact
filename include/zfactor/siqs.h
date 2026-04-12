#pragma once

// Self-Initializing Quadratic Sieve (SIQS).
//
// Factor a composite integer n using SIQS.  This is the method of choice
// for composites in the 80-120 digit range (~265-400 bits).  Below that,
// ECM/rho/SQUFOF are faster; above that, GNFS wins.
//
// Algorithm outline:
//   1. Select multiplier k (Knuth-Schroeppel) and compute kn = k*n.
//   2. Build factor base: primes p with (kn/p) ≥ 0, compute sqrt(kn) mod p.
//   3. Generate SIQS polynomials Q(x) = (Ax+B)^2 - kn.
//      Self-initialization: each A yields 2^(s-1) B-polynomials via Gray code.
//   4. Sieve Q(x) over [-M, M] using log-approximation byte array.
//   5. Trial divide candidates to find smooth relations (full + partial).
//   6. Collect enough relations (> FB size + surplus).
//   7. Block Lanczos over GF(2) to find null-space dependencies.
//   8. Square root phase: for each dependency, compute X² ≡ Y² (mod n)
//      and extract gcd(X-Y, n).
//
// Reference implementations: msieve (Papadopoulos), YAFU (Buhrow).

#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/iroot.h"
#include "zfactor/siqs/params.h"
#include "zfactor/siqs/factor_base.h"
#include "zfactor/siqs/poly.h"
#include "zfactor/siqs/sieve.h"
#include "zfactor/siqs/relation.h"
#include "zfactor/siqs/tdiv.h"
#include "zfactor/siqs/sqrt_phase.h"
#include "zfactor/siqs/lanczos.h"

namespace zfactor::siqs {

template<int N>
std::optional<fixint::UInt<N>> siqs(const fixint::UInt<N>& n) {
    // --- Step 1: Multiplier selection ---
    uint32_t k = select_multiplier<N>(n);

    // Compute kn = k * n
    fixint::UInt<N> kn;
    {
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) {
            unsigned __int128 w = (unsigned __int128)n.d[i] * k + carry;
            kn.d[i] = (uint64_t)w;
            carry = (uint64_t)(w >> 64);
        }
    }

    uint32_t bits = kn.bit_length();
    fprintf(stderr, "SIQS: n is %u bits, multiplier k=%u, kn is %u bits\n",
            n.bit_length(), k, bits);

    // --- Step 2: Parameters and factor base ---
    SiqsParams params = get_params(bits);
    fprintf(stderr, "SIQS: FB size=%u, LP mult=%u, blocks=%u\n",
            params.fb_size, params.lp_mult, params.num_blocks);

    FactorBase fb = FactorBase::build<N>(n, params, k);
    fprintf(stderr, "SIQS: built factor base with %u primes (max=%u)\n",
            fb.fb_size, fb.prime[fb.fb_size - 1]);

    // --- Step 3: Initialize sieve state ---
    SieveState sieve_state;
    sieve_state.init(params, fb);

    uint32_t sieve_len = sieve_state.sieve_len;
    uint32_t half_sieve = sieve_state.half_sieve;

    // Compute sieve threshold.
    // Q(x) ≈ M * sqrt(kn/2) at the edges, where M = half_sieve.
    // threshold = log2(Q_max) - log2(large_prime_max) - tolerance
    {
        // Q(x) ≈ sqrt(2*kn) * M / A at the sieve edges. But near x=0, Q ≈ C ≈ sqrt(kn).
        // We want candidates where Q(x) is smooth over FB ∪ {large primes up to LP_max}.
        // threshold = log2(Q_typical) - log2(LP_max) - fudge
        // where fudge accounts for logp rounding (~5 bits) and the fact that
        // candidates near x=0 have smaller Q(x).
        // A conservative approach: use the smaller of edge and center estimates.
        double log2_qedge = std::log2((double)half_sieve) + (double)bits / 2.0 - 0.5;
        double log2_qcenter = (double)bits / 2.0;  // Q(0) ≈ C ≈ sqrt(kn)
        double log2_lp = std::log2((double)fb.large_prime_max);
        // Use center estimate with generous fudge factor
        int thresh = (int)(log2_qcenter - log2_lp - 5);
        if (thresh < 5) thresh = 5;
        if (thresh > 240) thresh = 240;
        sieve_state.threshold = (uint8_t)thresh;
    }
    fprintf(stderr, "SIQS: sieve threshold=%u, sieve_len=%u\n",
            sieve_state.threshold, sieve_len);

    // --- Step 4: Sieve loop ---
    RelationSet rels;
    uint32_t target_relations = fb.fb_size + 64;

    // Polynomial storage for square root phase
    std::vector<fixint::UInt<N>> poly_a_list;
    std::vector<std::vector<fixint::UInt<N>>> poly_b_list;

    SiqsPoly<N> poly;
    uint32_t seed1 = 11111111, seed2 = 22222222;
    uint32_t total_polys = 0;
    uint32_t a_poly_count = 0;

    // Use only full relations for now (no SLP merging yet)
    auto have_enough_full = [&]() {
        return rels.num_full >= fb.fb_size + 64;
    };
    while (!have_enough_full()) {
        // Choose new A
        new_poly_a<N>(poly, fb, kn, sieve_len / 2, seed1, seed2);
        init_poly_b<N>(poly, fb, kn, sieve_len);

        uint32_t a_idx = a_poly_count++;
        poly_a_list.push_back(poly.A);
        poly_b_list.push_back({});

        // Sieve with all B-polynomials for this A
        do {
            // Store B BEFORE sieving so poly_b_list[a][poly_index] = current B
            while (poly_b_list[a_idx].size() <= (size_t)poly.poly_index)
                poly_b_list[a_idx].push_back({});
            poly_b_list[a_idx][poly.poly_index] = poly.B;

            // Fill buckets for large primes
            sieve_state.fill_buckets<N>(poly, fb);

            // Sieve each block
            uint32_t total_blocks = 2 * sieve_state.num_blocks;
            for (uint32_t blk = 0; blk < total_blocks; blk++) {
                sieve_state.sieve_block<N>(blk, poly, fb);

                // Scan and trial divide
                scan_and_tdiv<N>(blk, sieve_state.sieve, poly, fb, kn,
                                 half_sieve, a_idx, rels);
            }

            total_polys++;

            // Progress report every 100 polynomials
            if (total_polys % 100 == 0) {
                fprintf(stderr, "SIQS: %u polys, %u rels (%u full + %u SLP matched + %u DLP cycles) / %u needed\r",
                        total_polys, rels.effective_count(),
                        rels.num_full, rels.num_slp_matched, rels.dlp_cycles,
                        target_relations);
            }

        } while (next_poly_b<N>(poly, fb, kn));
    }

    fprintf(stderr, "\nSIQS: collected %u effective relations from %u polys\n",
            rels.effective_count(), total_polys);

    // --- Step 5: Build GF(2) matrix and run Block Lanczos ---

    // Build matrix: each relation is a column, each FB prime is a row.
    // Row 0 = sign, rows 1..fb_size-1 = FB primes.
    // For SLP/DLP, we add extra rows for large primes.

    // For simplicity, use only full relations + matched SLPs for now.
    // Build columns from full relations.
    std::vector<uint32_t> usable_rels;
    for (uint32_t i = 0; i < rels.relations.size(); i++) {
        auto& rel = rels.relations[i];
        if (rel.large_prime[0] == 1 && rel.large_prime[1] == 1) {
            usable_rels.push_back(i);
        }
    }

    // TODO: Properly merge matched SLP pairs into combined columns.
    // For now, only use full relations.

    uint32_t nrows = fb.fb_size;
    uint32_t ncols = (uint32_t)usable_rels.size();

    if (ncols <= nrows) {
        fprintf(stderr, "SIQS: not enough relations (%u cols, %u rows)\n", ncols, nrows);
        return std::nullopt;
    }

    fprintf(stderr, "SIQS: building %u x %u matrix\n", nrows, ncols);

    // Build la_col_t array for Block Lanczos
    LaCol* cols = (LaCol*)calloc(ncols, sizeof(LaCol));

    for (uint32_t c = 0; c < ncols; c++) {
        auto& rel = rels.relations[usable_rels[c]];

        // Count exponents mod 2 for each FB prime
        std::vector<uint32_t> odd_rows;

        if (rel.sign)
            odd_rows.push_back(0);  // sign row

        // Count FB prime exponents
        std::unordered_map<uint16_t, uint32_t> exp_count;
        for (auto fi : rel.fb_offsets)
            exp_count[fi]++;

        for (auto& [fi, cnt] : exp_count) {
            if (cnt & 1)
                odd_rows.push_back(fi);
        }

        std::sort(odd_rows.begin(), odd_rows.end());

        cols[c].weight = (uint32_t)odd_rows.size();
        cols[c].data = (uint32_t*)malloc(odd_rows.size() * sizeof(uint32_t));
        memcpy(cols[c].data, odd_rows.data(), odd_rows.size() * sizeof(uint32_t));
        cols[c].cycle.num_relations = 1;
        cols[c].cycle.list = (uint32_t*)malloc(sizeof(uint32_t));
        cols[c].cycle.list[0] = c;
    }

    // Run Block Lanczos
    uint32_t num_deps = 0;
    uint64_t* deps = block_lanczos(&nrows, 0, &ncols, cols, &num_deps);

    if (!deps || num_deps == 0) {
        fprintf(stderr, "SIQS: Block Lanczos failed to find dependencies\n");
        free(deps);
        return std::nullopt;
    }

    fprintf(stderr, "SIQS: found %u dependencies (%u reduced cols)\n", num_deps, ncols);

    // --- Step 6: Square root phase ---
    // Expand Lanczos dependencies from reduced columns back to original relations.
    // For each dependency j (0..num_deps-1), collect all original relation indices.
    // deps[c] bit j means reduced column c is in dependency j.
    // cols[c].cycle.list[0..num_relations-1] are original column indices.
    // usable_rels[orig_col] maps to the relation index in rels.relations.

    // First: build the full relation list
    std::vector<Relation>& all_rels = rels.relations;

    // For each dependency, build a list of relation indices
    std::vector<std::vector<uint32_t>> dep_rel_lists(num_deps);
    for (uint32_t c = 0; c < ncols; c++) {
        for (uint32_t j = 0; j < num_deps; j++) {
            if (!(deps[c] & ((uint64_t)1 << j))) continue;
            // This reduced column is in dependency j
            for (uint32_t r = 0; r < cols[c].cycle.num_relations; r++) {
                uint32_t orig_col = cols[c].cycle.list[r];
                if (orig_col < usable_rels.size()) {
                    dep_rel_lists[j].push_back(usable_rels[orig_col]);
                }
            }
        }
    }

    free_cols(cols, ncols);

    // Now try each dependency
    for (uint32_t dep = 0; dep < num_deps; dep++) {
        auto& rel_indices = dep_rel_lists[dep];
        if (rel_indices.empty()) continue;

        fixint::UInt<N> X = {}; X.d[0] = 1;
        std::vector<uint32_t> fb_counts(fb.fb_size, 0);

        for (uint32_t ri : rel_indices) {
            auto& rel = all_rels[ri];

            auto& A = poly_a_list[rel.a_poly_idx];
            fixint::UInt<N> B = {};
            if (rel.a_poly_idx < poly_b_list.size() &&
                rel.poly_idx < poly_b_list[rel.a_poly_idx].size()) {
                B = poly_b_list[rel.a_poly_idx][rel.poly_idx];
            }

            int32_t x = (int32_t)rel.sieve_offset - (int32_t)half_sieve;
            uint32_t abs_x = (uint32_t)std::abs(x);

            fixint::UInt<N> ax = {};
            {
                uint64_t carry = 0;
                for (int i = 0; i < N; i++) {
                    unsigned __int128 w = (unsigned __int128)A.d[i] * abs_x + carry;
                    ax.d[i] = (uint64_t)w;
                    carry = (uint64_t)(w >> 64);
                }
            }

            fixint::UInt<N> val;
            if (x >= 0) {
                fixint::mpn::add<N>(val.d, ax.d, B.d);
            } else {
                if (fixint::mpn::cmp<N>(B.d, ax.d) >= 0)
                    fixint::mpn::sub<N>(val.d, B.d, ax.d);
                else
                    fixint::mpn::sub<N>(val.d, ax.d, B.d);
            }

            if (fixint::mpn::cmp<N>(val.d, n.d) >= 0) {
                fixint::UInt<N> q, r;
                fixint::mpn::divrem<N>(q.d, r.d, val.d, n.d);
                val = r;
            }

            if (!val.is_zero())
                X = siqs::modmul<N>(X, val, n);

            for (auto fi : rel.fb_offsets)
                fb_counts[fi]++;

            // Track sign: if Q(x) < 0, the factored value is -|A*Q(x)|
            // We handle this by noting that the sign row ensures an even
            // number of negative values in each dependency, so signs cancel.
            // But we still need to include them in the exponent counting
            // for the matrix to work. The sign is already in fb_counts
            // via the matrix construction (row 0 = sign), so it's accounted
            // for automatically. No extra work needed here.
        }

        // Y = product of p^(count/2) mod n
        fixint::UInt<N> Y = {}; Y.d[0] = 1;
        for (uint32_t fi = 0; fi < fb.fb_size; fi++) {
            uint32_t count = fb_counts[fi];
            if (count < 2) continue;
            uint32_t exp = count / 2;
            uint32_t p = fb.prime[fi];
            if (p == 0) continue;
            fixint::UInt<N> base = {}; base.d[0] = p;
            fixint::UInt<N> pw = siqs::modpow<N>(base, exp, n);
            Y = siqs::modmul<N>(Y, pw, n);
        }

        // gcd(X-Y, n) and gcd(X+Y, n)
        fixint::UInt<N> diff, sum;
        if (fixint::mpn::cmp<N>(X.d, Y.d) >= 0)
            fixint::mpn::sub<N>(diff.d, X.d, Y.d);
        else
            fixint::mpn::sub<N>(diff.d, Y.d, X.d);

        fixint::mpn::add<N>(sum.d, X.d, Y.d);
        if (fixint::mpn::cmp<N>(sum.d, n.d) >= 0)
            fixint::mpn::sub<N>(sum.d, sum.d, n.d);

        auto check_factor = [&](fixint::UInt<N>& g) -> bool {
            if (g.is_zero()) return false;
            if (fixint::mpn::cmp<N>(g.d, n.d) == 0) return false;
            if (g.d[0] == 1 && g.bit_length() == 1) return false;
            if (k > 1) {
                uint64_t d = fixint::gcd_u64(k, uint_mod_u32<N>(g, k));
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

        auto g1 = fixint::gcd<N>(diff, n);
        if (check_factor(g1)) {
            free(deps);
            fprintf(stderr, "SIQS: found factor from dependency %u!\n", dep);
            return g1;
        }
        auto g2 = fixint::gcd<N>(sum, n);
        if (check_factor(g2)) {
            free(deps);
            fprintf(stderr, "SIQS: found factor from dependency %u!\n", dep);
            return g2;
        }
    }

    free(deps);
    fprintf(stderr, "SIQS: sqrt phase failed to find non-trivial factor\n");
    return std::nullopt;
}

} // namespace zfactor::siqs
