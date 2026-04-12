// ECM stage 2: FFT polynomial evaluation via FLINT.
//
// Builds f(y) = ∏(y - y_j) from baby-step y-coordinates, evaluates at
// giant-step y-coordinates using FLINT's subproduct tree.  Cost is
// O((NJ + NG) · log²(max(NJ,NG))) vs O(NJ · NG) for the BSGS inner loop.
//
// Uses y-coordinate (not x) because on twisted Edwards a=-1 the negation
// is (-x, y), so y(P) = y(-P).  This detects both p = m·P + j and
// p = m·P - j in a single pass, covering all primes in (B1, B2].
//
// The primorial P is chosen dynamically: φ(P) ≈ (B2-B1)/P balances
// the polynomial degree (NJ = φ(P)/2) against evaluation points (NG).
//
// Requires FLINT: compile with -DZFACTOR_USE_FLINT and link -lflint -lgmp.
#pragma once

#include <gmp.h>
#include <fmpz.h>
#include <fmpz_vec.h>
#include <fmpz_mod.h>
#include <fmpz_mod_poly.h>

#include "zfactor/edwards.h"
#include "zfactor/fixint/gcd.h"
#include "zfactor/eecm/stage2.h"   // batch_invert, ed_scalar_u64, Stage2Result

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace zfactor::ecm {

namespace detail_s2poly {

// ── Primorial table ──

struct Primorial { uint64_t P, phi; int np; };

// P = product of first np primes, phi = φ(P).
constexpr Primorial PRIMORIALS[] = {
    {         6,        2, 2},   // 2·3
    {        30,        8, 3},   // 2·3·5
    {       210,       48, 4},   // ×7
    {      2310,      480, 5},   // ×11
    {     30030,     5760, 6},   // ×13
    {    510510,    92160, 7},   // ×17
    {   9699690,  1658880, 8},   // ×19
};
constexpr int NPRIM = 7;
constexpr int SPRIMES[] = {2, 3, 5, 7, 11, 13, 17, 19};

// Choose P such that φ(P) ≈ range/P  ⟺  φ(P)·P ≈ range.
inline Primorial choose_primorial(uint64_t range) {
    int best = 0;
    double best_err = 1e30;
    double lr = std::log((double)range);
    for (int i = 0; i < NPRIM; ++i) {
        double ng = (double)range / PRIMORIALS[i].P;
        if (ng < 2) continue;
        double lpp = std::log((double)PRIMORIALS[i].phi * PRIMORIALS[i].P);
        double err = std::abs(lpp - lr);
        if (err < best_err) { best_err = err; best = i; }
    }
    return PRIMORIALS[best];
}

// ── Coprime residue cache (thread-local) ──

struct CoprimeInfo {
    uint64_t P = 0;
    std::vector<int> res;     // coprime j ∈ [1, P/2), sorted
    std::vector<int> gaps;    // gaps[0] = res[0]; gaps[i] = res[i]-res[i-1]
    int max_gap = 0;
    std::vector<bool> gap_used;
};

inline const CoprimeInfo& get_coprime_info(uint64_t P, int np) {
    static thread_local CoprimeInfo cache;
    if (cache.P == P) return cache;
    cache.P = P;
    cache.res.clear();
    int half = (int)(P / 2);
    for (int j = 1; j < half; ++j) {
        bool ok = true;
        for (int k = 0; k < np; ++k)
            if (j % SPRIMES[k] == 0) { ok = false; break; }
        if (ok) cache.res.push_back(j);
    }
    int NJ = (int)cache.res.size();
    cache.gaps.resize(NJ);
    cache.gaps[0] = cache.res[0];
    cache.max_gap = cache.gaps[0];
    for (int i = 1; i < NJ; ++i) {
        cache.gaps[i] = cache.res[i] - cache.res[i - 1];
        cache.max_gap = std::max(cache.max_gap, cache.gaps[i]);
    }
    cache.gap_used.assign(cache.max_gap + 1, false);
    for (auto g : cache.gaps) cache.gap_used[g] = true;
    return cache;
}

}  // namespace detail_s2poly


// ── UInt ↔ fmpz conversion ──

template<int N>
inline void uint_to_fmpz(fmpz_t out, const fixint::UInt<N>& x) {
    mpz_t z;
    mpz_init2(z, N * 64);
    mpz_import(z, N, -1, 8, 0, 0, x.d);
    fmpz_set_mpz(out, z);
    mpz_clear(z);
}

template<int N>
inline void fmpz_to_uint(fixint::UInt<N>& out, const fmpz_t x) {
    mpz_t z;
    mpz_init(z);
    fmpz_get_mpz(z, x);
    std::memset(out.d, 0, sizeof(out.d));
    size_t cnt;
    mpz_export(out.d, &cnt, -1, 8, 0, 0, z);
    mpz_clear(z);
}


// ── FFT polynomial stage 2 ──

template<int N>
Stage2Result<N> ecm_stage2_poly(const EdPoint<N>& Q,
                                const EdCurve<N>& curve,
                                uint64_t B1, uint64_t B2,
                                const fixint::MontCtx<N>& ctx) {
    Stage2Result<N> result;
    if (B2 <= B1) return result;

    using namespace detail_s2poly;
    auto m = edwards_mont_ops<N>(ctx);

    // ─── 1. Choose primorial, get coprime residues ───
    uint64_t range = B2 - B1;
    auto pr = choose_primorial(range);
    uint64_t P = pr.P;
    const auto& ci = get_coprime_info(P, pr.np);
    int NJ = (int)ci.res.size();   // = φ(P)/2

    // ─── 2. Baby steps: gap-based walk ───
    // Precompute [g]*Q for each distinct gap between coprime residues.
    std::vector<EdPoint<N>> gap_pt(ci.max_gap + 1);
    for (int g = 1; g <= ci.max_gap; ++g)
        if (ci.gap_used[g])
            gap_pt[g] = ed_scalar_u64<N>(Q, (uint64_t)g, curve, ctx);

    // Walk: baby[0] = [res[0]]*Q, baby[i] = baby[i-1] + [gap_i]*Q.
    std::vector<EdPoint<N>> baby(NJ);
    baby[0] = gap_pt[ci.gaps[0]];
    for (int i = 1; i < NJ; ++i)
        baby[i] = ed_add<N>(baby[i - 1], gap_pt[ci.gaps[i]], curve, ctx);
    gap_pt.clear();

    // Batch invert Z → affine y = Y/Z.
    // (y-coordinate, not x: on twisted Edwards -P = (-x,y), so y(P)=y(-P).)
    std::vector<fixint::UInt<N>> baby_y(NJ);
    {
        std::vector<fixint::UInt<N>> zv(NJ);
        for (int i = 0; i < NJ; ++i) zv[i] = baby[i].Z;
        auto f = batch_invert<N>(zv.data(), NJ, ctx);
        if (f) { result.factor_found = true; result.factor = *f; return result; }
        for (int i = 0; i < NJ; ++i)
            baby_y[i] = m.mul(baby[i].Y, zv[i]);
    }
    baby.clear();

    // ─── 3. Giant steps ───
    uint64_t m0 = (B1 + P) / P;
    uint64_t mK = B2 / P;
    if (mK < m0) return result;
    int NG = (int)(mK - m0 + 1);

    auto dQ = ed_scalar_u64<N>(Q, P, curve, ctx);
    auto R = ed_scalar_u64<N>(Q, m0 * P, curve, ctx);

    std::vector<fixint::UInt<N>> giant_Y(NG), giant_Z(NG);
    for (int k = 0; k < NG; ++k) {
        giant_Y[k] = R.Y;
        giant_Z[k] = R.Z;
        if (k + 1 < NG) R = ed_add<N>(R, dQ, curve, ctx);
    }

    // Batch invert giant Z → affine y.
    {
        auto f = batch_invert<N>(giant_Z.data(), NG, ctx);
        if (f) { result.factor_found = true; result.factor = *f; return result; }
        for (int k = 0; k < NG; ++k)
            giant_Y[k] = m.mul(giant_Y[k], giant_Z[k]);
        // giant_Y[k] is now the affine y-coordinate in Montgomery form.
    }
    giant_Z.clear();

    // ─── 4. FLINT: build polynomial from baby roots ───
    fmpz_t n_fz;
    fmpz_init(n_fz);
    uint_to_fmpz<N>(n_fz, ctx.mod);

    fmpz_mod_ctx_t fctx;
    fmpz_mod_ctx_init(fctx, n_fz);

    // Baby y-values as fmpz roots (drop Montgomery form).
    fmpz* roots = _fmpz_vec_init(NJ);
    {
        fmpz_t tmp;
        fmpz_init(tmp);
        for (int j = 0; j < NJ; ++j) {
            uint_to_fmpz<N>(tmp, m.drop(baby_y[j]));
            fmpz_set(roots + j, tmp);
        }
        fmpz_clear(tmp);
    }
    baby_y.clear();

    // f(y) = ∏(y - roots[j])  via FLINT's balanced product tree.
    fmpz_mod_poly_t poly;
    fmpz_mod_poly_init(poly, fctx);
    fmpz_mod_poly_product_roots_fmpz_vec(poly, roots, NJ, fctx);
    _fmpz_vec_clear(roots, NJ);

    // Giant y-values as fmpz evaluation points.
    fmpz* xs = _fmpz_vec_init(NG);
    fmpz* ys = _fmpz_vec_init(NG);
    {
        fmpz_t tmp;
        fmpz_init(tmp);
        for (int k = 0; k < NG; ++k) {
            uint_to_fmpz<N>(tmp, m.drop(giant_Y[k]));
            fmpz_set(xs + k, tmp);
        }
        fmpz_clear(tmp);
    }
    giant_Y.clear();

    // ─── 5. Multi-point evaluation ───
    fmpz_mod_poly_evaluate_fmpz_vec(ys, poly, xs, NG, fctx);

    // ─── 6. Accumulate product, batched gcd ───
    {
        fmpz_t acc, g;
        fmpz_init(acc);
        fmpz_init(g);
        fmpz_one(acc);
        constexpr int BATCH = 128;

        for (int k = 0; k < NG; ++k) {
            fmpz_mul(acc, acc, ys + k);
            fmpz_mod(acc, acc, n_fz);

            if (k % BATCH == BATCH - 1 || k == NG - 1) {
                if (fmpz_is_zero(acc)) {
                    // Product hit 0 — check individuals in this batch.
                    int s = k - (k % BATCH);
                    for (int i = s; i <= k; ++i) {
                        fmpz_gcd(g, ys + i, n_fz);
                        if (fmpz_cmp_ui(g, 1) > 0 && fmpz_cmp(g, n_fz) < 0) {
                            result.factor_found = true;
                            fmpz_to_uint<N>(result.factor, g);
                            goto done;
                        }
                    }
                } else {
                    fmpz_gcd(g, acc, n_fz);
                    if (fmpz_cmp_ui(g, 1) > 0 && fmpz_cmp(g, n_fz) < 0) {
                        result.factor_found = true;
                        fmpz_to_uint<N>(result.factor, g);
                        goto done;
                    }
                }
                fmpz_one(acc);
            }
        }
    done:
        fmpz_clear(acc);
        fmpz_clear(g);
    }

    // ─── Cleanup ───
    _fmpz_vec_clear(xs, NG);
    _fmpz_vec_clear(ys, NG);
    fmpz_mod_poly_clear(poly, fctx);
    fmpz_mod_ctx_clear(fctx);
    fmpz_clear(n_fz);

    return result;
}

}  // namespace zfactor::ecm
