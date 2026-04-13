// test_zint_ntt_regress.cpp
//
// Focused regression harness for large zint NTT multiplies against GMP.
// It exercises:
// - current dispatch (multi-engine p30x3 + p50x4 fallback)
// - recreated pre-change p30x3 path from the clean zint submodule state
// - forced EngineA / EngineB / EngineC
// - forced p50x4
//
// Input generation matches bench_ntt_vs_gmp.cpp: xoshiro256++ with the top
// limb forced non-zero.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "zint/zint.hpp"
#include "zint/rng.hpp"
#include "zint/ntt/api.hpp"
#include "zint/ntt/p30x3/crt.hpp"
#include "zint/ntt/p30x3/engines.hpp"

#include <gmp.h>
namespace gmp {
    inline mp_limb_t mul(mp_limb_t* rp, const mp_limb_t* ap, mp_size_t an,
                         const mp_limb_t* bp, mp_size_t bn) {
        return __gmpn_mul(rp, ap, an, bp, bn);
    }
}
#undef mpn_mul
#undef mpn_mul_n

using u32 = zint::ntt::u32;
using u64 = zint::ntt::u64;
using idt = zint::ntt::idt;
static_assert(sizeof(u64) == sizeof(mp_limb_t), "zint and GMP limb widths must match");

namespace old_api {

inline void big_multiply(
    u32* out, idt out_len,
    const u32* a, idt na,
    const u32* b, idt nb)
{
    using namespace zint::ntt;
    using B = Avx2;
    using Vec = typename B::Vec;

    const idt min_len = na + nb;
    idt N = ceil_smooth(min_len > 64 ? min_len : 64);
    idt ntt_vecs = N / B::LANES;
    if (ntt_vecs < 8) {
        ntt_vecs = 8;
        N = ntt_vecs * B::LANES;
    }

    ::zint::ScratchScope scope(::zint::scratch());
    Vec* buf = scope.alloc<Vec>(4 * ntt_vecs, 64);
    Vec* f0 = buf + 0 * ntt_vecs;
    Vec* f1 = buf + 1 * ntt_vecs;
    Vec* f2 = buf + 2 * ntt_vecs;
    Vec* g = buf + 3 * ntt_vecs;

    ntt_conv_one_prime<B, CRT_P0>((u32*)f0, (u32*)g, ntt_vecs, a, na, b, nb, N);
    ntt_conv_one_prime<B, CRT_P1>((u32*)f1, (u32*)g, ntt_vecs, a, na, b, nb, N);
    ntt_conv_one_prime<B, CRT_P2>((u32*)f2, (u32*)g, ntt_vecs, a, na, b, nb, N);

    const idt result_len = (std::min)(min_len, out_len);
    crt_and_propagate(out, result_len, (u32*)f0, (u32*)f1, (u32*)f2);
}

inline void big_multiply_u64(
    u64* out, idt out_len,
    const u64* a, idt na,
    const u64* b, idt nb)
{
    using namespace zint::ntt;
    const idt na32 = 2 * na;
    const idt nb32 = 2 * nb;
    const idt out32 = 2 * out_len;
    const idt ntt_size = ceil_smooth(na32 + nb32 > 64 ? na32 + nb32 : 64);

    if (ntt_size <= 12582912) {
        big_multiply((u32*)out, out32, (const u32*)a, na32, (const u32*)b, nb32);
    } else {
        p50x4::Ntt4& engine = p50x4::Ntt4::instance();
        engine.multiply(out, (std::size_t)out_len,
                        a, (std::size_t)na,
                        b, (std::size_t)nb);
    }
}

} // namespace old_api

enum class Mode {
    current_dispatch,
    old_dispatch,
    engine_a,
    engine_b,
    engine_c,
    p50x4_only,
};

static const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::current_dispatch: return "current_dispatch";
        case Mode::old_dispatch:     return "old_dispatch";
        case Mode::engine_a:         return "force_engine_a";
        case Mode::engine_b:         return "force_engine_b";
        case Mode::engine_c:         return "force_engine_c";
        case Mode::p50x4_only:       return "force_p50x4";
        default:                     return "unknown";
    }
}

static void fill_random(u64* dst, std::size_t n, zint::xoshiro256pp& rng) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = rng.next();
    }
    if (n) {
        dst[n - 1] |= (u64(1) << 63);
    }
}

static u32 reduce_input_mod2p(u32 x, u32 mod) {
    const u64 mod2 = 2ULL * mod;
    const u64 mod4 = 4ULL * mod;
    const u64 mod8 = 8ULL * mod;
    if (mod8 <= 0xffffffffULL && x >= mod8) x -= (u32)mod8;
    if (mod4 <= 0xffffffffULL && x >= mod4) x -= (u32)mod4;
    if (x >= mod2) x -= (u32)mod2;
    return x;
}

static u32 reduce_input_modp(u32 x, u32 mod) {
    x = reduce_input_mod2p(x, mod);
    if (x >= mod) x -= mod;
    return x;
}

static std::vector<u32> to_u32_limbs(const std::vector<u64>& x) {
    std::vector<u32> out(2 * x.size());
    std::memcpy(out.data(), x.data(), x.size() * sizeof(u64));
    return out;
}

static bool fill_case_from_bench_sweep(std::size_t target_na,
                                       std::size_t target_nb,
                                       std::vector<u64>& a,
                                       std::vector<u64>& b) {
    static const std::size_t sizes[] = {
        2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
        1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768,
        49152, 65536, 98304, 131072, 196608, 262144, 393216, 524288,
        786432, 1048576, 1572864, 2097152, 3145728, 4194304, 6291456,
        8388608, 10485760,
    };

    zint::xoshiro256pp rng(0xA5A55A5A12345678ULL);
    for (std::size_t s : sizes) {
        {
            std::vector<u64> ca(s), cb(s);
            fill_random(ca.data(), ca.size(), rng);
            fill_random(cb.data(), cb.size(), rng);
            if (s == target_na && s == target_nb) {
                a = std::move(ca);
                b = std::move(cb);
                return true;
            }
        }
        {
            const std::size_t s2 = s > 1 ? s / 2 : 1;
            std::vector<u64> ca(s), cb(s2);
            fill_random(ca.data(), ca.size(), rng);
            fill_random(cb.data(), cb.size(), rng);
            if (s == target_na && s2 == target_nb) {
                a = std::move(ca);
                b = std::move(cb);
                return true;
            }
        }
    }
    return false;
}

static void gmp_mul_u64(std::vector<u64>& out,
                        const std::vector<u64>& a,
                        const std::vector<u64>& b) {
    std::fill(out.begin(), out.end(), 0);
    if (a.size() >= b.size()) {
        gmp::mul((mp_limb_t*)out.data(),
                 (const mp_limb_t*)a.data(), (mp_size_t)a.size(),
                 (const mp_limb_t*)b.data(), (mp_size_t)b.size());
    } else {
        gmp::mul((mp_limb_t*)out.data(),
                 (const mp_limb_t*)b.data(), (mp_size_t)b.size(),
                 (const mp_limb_t*)a.data(), (mp_size_t)a.size());
    }
}

struct RangeScan {
    bool ok;
    idt bad_idx;
    u32 bad_value;
    u32 max_value;
};

static RangeScan scan_upper_bound(const u32* data, idt n, u32 bound) {
    RangeScan rs{true, 0, 0, 0};
    for (idt i = 0; i < n; ++i) {
        const u32 v = data[i];
        if (v > rs.max_value) {
            rs.max_value = v;
        }
        if (v >= bound) {
            rs.ok = false;
            rs.bad_idx = i;
            rs.bad_value = v;
            break;
        }
    }
    return rs;
}

static bool print_range_check(const char* label,
                              const u32* data,
                              idt n,
                              u32 bound) {
    const RangeScan rs = scan_upper_bound(data, n, bound);
    if (!rs.ok) {
        std::printf("  %-28s FAIL idx=%zu val=%u bound=%u max_before_fail=%u\n",
                    label,
                    (std::size_t)rs.bad_idx,
                    rs.bad_value,
                    bound,
                    rs.max_value);
        return false;
    }
    std::printf("  %-28s PASS max=%u bound=%u\n",
                label, rs.max_value, bound);
    return true;
}

static u32 add2_scalar(u32 a, u32 b, u32 mod2) {
    u32 x = a + b;
    return (x >= mod2) ? (x - mod2) : x;
}

static u32 sub2_scalar(u32 a, u32 b, u32 mod2) {
    return (a >= b) ? (a - b) : (a + mod2 - b);
}

static u32 lazy_sub_scalar(u32 a, u32 b, u32 mod2) {
    return a + (mod2 - b);
}

static u32 lazy_add_scalar(u32 a, u32 b) {
    return a + b;
}

template<u32 Mod>
static bool first_radix4_notw_compare(const std::vector<u32>& src,
                                      idt src_len,
                                      idt N_u32,
                                      idt outer_sub,
                                      const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using Vec = typename B::Vec;

    const idt ntt_vecs = N_u32 / B::LANES;
    const int k = ntt_ctzll(ntt_vecs);
    const idt m = ntt_vecs >> k;
    if (m != 5 || outer_sub >= m) {
        std::printf("  %s SKIP\n", label);
        return true;
    }

    const idt sub_n = idt{1} << k;
    const idt L = sub_n >> 2;
    const MontScalar ms{Mod};
    const RootPlan<Mod> roots{};
    const MontVec<B> mv(ms.mod, ms.niv, roots.img);
    const u32 mod2 = ms.mod2;

    ::zint::ScratchScope scope(::zint::scratch());
    Vec* buf = scope.alloc<Vec>(ntt_vecs + sub_n, 64);
    Vec* simd_sub = buf + ntt_vecs;
    reduce_and_pad<B, Mod>((u32*)buf, src.data(), src_len, N_u32);
    Radix5Kernel<B, Mod>::dif_pass(buf, ntt_vecs, mv, roots);
    std::memcpy(simd_sub, buf + outer_sub * sub_n, sub_n * sizeof(Vec));

    std::vector<u32> scalar((u32*)simd_sub, (u32*)simd_sub + sub_n * B::LANES);

    Radix4Kernel<B>::dif_butterfly_notw(simd_sub, simd_sub + L, simd_sub + 2 * L, simd_sub + 3 * L, L, mv);

    auto idx = [&](idt vec_index, int lane) -> idt {
        return vec_index * B::LANES + lane;
    };
    auto mul_by_img_scalar = [&](u32 x) -> u32 {
        return ms.reduce((u64)x * roots.img);
    };

    for (idt i = 0; i < L; ++i) {
        for (int lane = 0; lane < B::LANES; ++lane) {
            const u32 f0 = scalar[idx(i, lane)];
            const u32 f1 = scalar[idx(i + L, lane)];
            const u32 f2 = scalar[idx(i + 2 * L, lane)];
            const u32 f3 = scalar[idx(i + 3 * L, lane)];

            const u32 g1 = add2_scalar(f1, f3, mod2);
            const u32 g3 = mul_by_img_scalar(lazy_sub_scalar(f1, f3, mod2));
            const u32 g0 = add2_scalar(f0, f2, mod2);
            const u32 g2 = sub2_scalar(f0, f2, mod2);

            scalar[idx(i, lane)] = add2_scalar(g0, g1, mod2);
            scalar[idx(i + L, lane)] = lazy_sub_scalar(g0, g1, mod2);
            scalar[idx(i + 2 * L, lane)] = lazy_add_scalar(g2, g3);
            scalar[idx(i + 3 * L, lane)] = lazy_sub_scalar(g2, g3, mod2);
        }
    }

    const u32* simd = (const u32*)simd_sub;
    for (idt i = 0; i < sub_n * B::LANES; ++i) {
        if (simd[i] != scalar[i]) {
            std::printf("  %s FAIL idx=%zu simd=%u scalar=%u\n",
                        label, (std::size_t)i, simd[i], scalar[i]);
            return false;
        }
    }

    std::printf("  %s PASS\n", label);
    return true;
}

template<u32 Mod>
static void radix5_dif_scalar_pass(u32* f, idt ntt_vecs) {
    using namespace zint::ntt;
    const MontScalar ms{Mod};
    const RootPlan<Mod> roots{};
    const idt sub_n = ntt_vecs / 5;
    const int k = ntt_ctzll(sub_n);
    const u32 mod2 = ms.mod2;

    const u32 c1h = roots.c1h;
    const u32 c2h = roots.c2h;
    const u32 j1h = roots.j1h;
    const u32 j2h = roots.j2h;
    const u32 c12h = ms.shrink(roots.c1h + roots.c2h);
    const u32 j12s = ms.shrink(roots.j1h + roots.j2h);

    const u32 tw_root = roots.tw5_root[k];
    const u32 tw2_root = ms.mul_s(tw_root, tw_root);
    const u32 tw3_root = ms.mul_s(tw2_root, tw_root);
    const u32 tw4_root = ms.mul_s(tw2_root, tw2_root);

    auto mont_mul = [&](u32 x, u32 y) -> u32 {
        return ms.reduce((u64)x * y);
    };

    auto idx = [&](idt vec_index, int lane) -> idt {
        return vec_index * 8 + lane;
    };

    {
        const idt j = 0;
        for (int lane = 0; lane < 8; ++lane) {
            const u32 a = f[idx(j, lane)];
            const u32 b = f[idx(j + sub_n, lane)];
            const u32 c = f[idx(j + 2 * sub_n, lane)];
            const u32 d = f[idx(j + 3 * sub_n, lane)];
            const u32 e = f[idx(j + 4 * sub_n, lane)];

            const u32 s1 = add2_scalar(b, e, mod2);
            const u32 t1 = sub2_scalar(b, e, mod2);
            const u32 s2 = add2_scalar(c, d, mod2);
            const u32 t2 = sub2_scalar(c, d, mod2);

            const u32 f0 = add2_scalar(a, add2_scalar(s1, s2, mod2), mod2);

            const u32 p1 = mont_mul(s1, c1h);
            const u32 p2 = mont_mul(s2, c2h);
            const u32 p3 = mont_mul(add2_scalar(s1, s2, mod2), c12h);
            const u32 pp = add2_scalar(p1, p2, mod2);
            const u32 alpha = add2_scalar(a, pp, mod2);
            const u32 gamma = add2_scalar(a, sub2_scalar(p3, pp, mod2), mod2);

            const u32 q1 = mont_mul(t1, j1h);
            const u32 q2 = mont_mul(t2, j2h);
            const u32 q3 = mont_mul(sub2_scalar(t1, t2, mod2), j12s);
            const u32 beta = add2_scalar(q1, q2, mod2);
            const u32 delta = add2_scalar(sub2_scalar(q3, q1, mod2), q2, mod2);

            f[idx(j, lane)] = f0;
            f[idx(j + sub_n, lane)] = add2_scalar(alpha, beta, mod2);
            f[idx(j + 2 * sub_n, lane)] = add2_scalar(gamma, delta, mod2);
            f[idx(j + 3 * sub_n, lane)] = sub2_scalar(gamma, delta, mod2);
            f[idx(j + 4 * sub_n, lane)] = sub2_scalar(alpha, beta, mod2);
        }
    }

    u32 tw1 = tw_root;
    u32 tw2 = tw2_root;
    u32 tw3 = tw3_root;
    u32 tw4 = tw4_root;

    for (idt j = 1; j < sub_n; ++j) {
        for (int lane = 0; lane < 8; ++lane) {
            const u32 a = f[idx(j, lane)];
            const u32 b = f[idx(j + sub_n, lane)];
            const u32 c = f[idx(j + 2 * sub_n, lane)];
            const u32 d = f[idx(j + 3 * sub_n, lane)];
            const u32 e = f[idx(j + 4 * sub_n, lane)];

            const u32 s1 = add2_scalar(b, e, mod2);
            const u32 t1 = sub2_scalar(b, e, mod2);
            const u32 s2 = add2_scalar(c, d, mod2);
            const u32 t2 = sub2_scalar(c, d, mod2);

            const u32 f0 = add2_scalar(a, add2_scalar(s1, s2, mod2), mod2);

            const u32 p1 = mont_mul(s1, c1h);
            const u32 p2 = mont_mul(s2, c2h);
            const u32 p3 = mont_mul(add2_scalar(s1, s2, mod2), c12h);
            const u32 pp = add2_scalar(p1, p2, mod2);
            const u32 alpha = add2_scalar(a, pp, mod2);
            const u32 gamma = add2_scalar(a, sub2_scalar(p3, pp, mod2), mod2);

            const u32 q1 = mont_mul(t1, j1h);
            const u32 q2 = mont_mul(t2, j2h);
            const u32 q3 = mont_mul(sub2_scalar(t1, t2, mod2), j12s);
            const u32 beta = add2_scalar(q1, q2, mod2);
            const u32 delta = add2_scalar(sub2_scalar(q3, q1, mod2), q2, mod2);

            f[idx(j, lane)] = f0;
            f[idx(j + sub_n, lane)] = mont_mul(alpha + beta, tw1);
            f[idx(j + 2 * sub_n, lane)] = mont_mul(gamma + delta, tw2);
            f[idx(j + 3 * sub_n, lane)] = mont_mul(lazy_sub_scalar(gamma, delta, mod2), tw3);
            f[idx(j + 4 * sub_n, lane)] = mont_mul(lazy_sub_scalar(alpha, beta, mod2), tw4);
        }

        tw1 = ms.mul_s(tw1, tw_root);
        tw2 = ms.mul_s(tw2, tw2_root);
        tw3 = ms.mul_s(tw3, tw3_root);
        tw4 = ms.mul_s(tw4, tw4_root);
    }
}

template<u32 Mod>
static bool radix5_dif_scalar_compare(const std::vector<u32>& src,
                                      idt N_u32,
                                      const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using Vec = typename B::Vec;

    const idt ntt_vecs = N_u32 / B::LANES;
    if (ntt_vecs % 5 != 0) {
        std::printf("  %s SKIP (not 5*2^k)\n", label);
        return true;
    }

    ::zint::ScratchScope scope(::zint::scratch());
    Vec* buf = scope.alloc<Vec>(2 * ntt_vecs, 64);
    Vec* simd_buf = buf;
    Vec* scalar_seed = buf + ntt_vecs;
    reduce_and_pad<B, Mod>((u32*)simd_buf, src.data(), (idt)src.size(), N_u32);
    std::memcpy((u32*)scalar_seed, (u32*)simd_buf, N_u32 * sizeof(u32));

    std::vector<u32> simd(N_u32);
    std::vector<u32> scalar((u32*)buf, (u32*)buf + N_u32);

    {
        const RootPlan<Mod> roots{};
        const MontScalar ms{Mod};
        const MontVec<B> mv(ms.mod, ms.niv, roots.img);
        Radix5Kernel<B, Mod>::dif_pass(simd_buf, ntt_vecs, mv, roots);
    }
    std::memcpy(simd.data(), (u32*)simd_buf, N_u32 * sizeof(u32));
    scalar.assign((u32*)scalar_seed, (u32*)scalar_seed + N_u32);
    radix5_dif_scalar_pass<Mod>(scalar.data(), ntt_vecs);

    for (idt i = 0; i < N_u32; ++i) {
        if (simd[i] != scalar[i]) {
            std::printf("  %s FAIL idx=%zu simd=%u scalar=%u\n",
                        label, (std::size_t)i, simd[i], scalar[i]);
            return false;
        }
    }

    std::printf("  %s PASS\n", label);
    return true;
}

template<u32 Mod>
static bool full_roundtrip_check(const std::vector<u32>& src, idt N_u32, const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using S = NTTScheduler<B, Mod>;
    using Vec = typename B::Vec;

    const idt ntt_vecs = N_u32 / B::LANES;
    ::zint::ScratchScope scope(::zint::scratch());
    Vec* buf = scope.alloc<Vec>(ntt_vecs, 64);
    u32* data = (u32*)buf;

    reduce_and_pad<B, Mod>(data, src.data(), (idt)src.size(), N_u32);
    std::vector<u32> expect(N_u32, 0);
    for (idt i = 0; i < (idt)src.size(); ++i) {
        expect[i] = reduce_input_modp(src[i], Mod);
    }
    S::forward(buf, ntt_vecs);
    S::inverse(buf, ntt_vecs);

    for (idt i = 0; i < N_u32; ++i) {
        if (data[i] != expect[i]) {
            std::printf("  %s FAIL at u32=%zu got=%u expect=%u\n",
                        label, (std::size_t)i, data[i], expect[i]);
            return false;
        }
    }
    std::printf("  %s PASS\n", label);
    return true;
}

template<u32 Mod>
static bool b2_roundtrip_check(const std::vector<u32>& src, int lg_vecs, const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using S = NTTScheduler<B, Mod>;
    using Vec = typename B::Vec;

    const idt ntt_vecs = idt{1} << lg_vecs;
    const idt N_u32 = ntt_vecs * B::LANES;
    ::zint::ScratchScope scope(::zint::scratch());
    Vec* buf = scope.alloc<Vec>(ntt_vecs, 64);
    u32* data = (u32*)buf;

    reduce_and_pad<B, Mod>(data, src.data(), (idt)std::min<std::size_t>(src.size(), (std::size_t)N_u32), N_u32);
    std::vector<u32> expect(N_u32, 0);
    for (idt i = 0; i < (idt)std::min<std::size_t>(src.size(), (std::size_t)N_u32); ++i) {
        expect[i] = reduce_input_modp(src[i], Mod);
    }
    S::fwd_b2(buf, ntt_vecs);
    S::inv_b2(buf, ntt_vecs);

    for (idt i = 0; i < N_u32; ++i) {
        if (data[i] != expect[i]) {
            std::printf("  %s FAIL at u32=%zu got=%u expect=%u\n",
                        label, (std::size_t)i, data[i], expect[i]);
            return false;
        }
    }
    std::printf("  %s PASS\n", label);
    return true;
}

template<u32 Mod>
static bool major_phase_range_audit(const std::vector<u32>& a32,
                                    const std::vector<u32>& b32,
                                    idt N_u32,
                                    const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using S = NTTScheduler<B, Mod>;
    using Vec = typename B::Vec;

    const idt ntt_vecs = N_u32 / B::LANES;
    const int k = ntt_ctzll(ntt_vecs);
    const idt m = ntt_vecs >> k;
    const idt sub_n = idt{1} << k;
    const auto& roots = S::get_roots();
    const auto& ms = S::get_ms();
    const MontVec<B> mv(ms.mod, ms.niv, roots.img);

    ::zint::ScratchScope scope(::zint::scratch());
    Vec* f = scope.alloc<Vec>(ntt_vecs, 64);
    Vec* g = scope.alloc<Vec>(ntt_vecs, 64);

    const u32 bound_mod = ms.mod;
    const u32 bound_mod2 = ms.mod2;
    const u32 bound_mod4 = ms.mod2 * 2u;

    std::printf("  %s\n", label);

    reduce_and_pad<B, Mod>((u32*)f, a32.data(), (idt)a32.size(), N_u32);
    reduce_and_pad<B, Mod>((u32*)g, b32.data(), (idt)b32.size(), N_u32);
    bool ok = true;
    ok &= print_range_check("reduce/pad A", (const u32*)f, N_u32, bound_mod2);
    ok &= print_range_check("reduce/pad B", (const u32*)g, N_u32, bound_mod2);

    if (m == 3) {
        Radix3Kernel<B, Mod>::dif_pass(f, ntt_vecs, mv, roots);
        Radix3Kernel<B, Mod>::dif_pass(g, ntt_vecs, mv, roots);
    } else if (m == 5) {
        Radix5Kernel<B, Mod>::dif_pass(f, ntt_vecs, mv, roots);
        Radix5Kernel<B, Mod>::dif_pass(g, ntt_vecs, mv, roots);
    }
    if (m != 1) {
        ok &= print_range_check("outer DIF A", (const u32*)f, N_u32, bound_mod2);
        ok &= print_range_check("outer DIF B", (const u32*)g, N_u32, bound_mod2);
    }

    for (idt sub = 0; sub < m; ++sub) {
        S::fwd_b2(f + sub * sub_n, sub_n);
        S::fwd_b2(g + sub * sub_n, sub_n);
        char label_a[64];
        char label_b[64];
        std::snprintf(label_a, sizeof(label_a), "fwd_b2 A sub=%zu", (std::size_t)sub);
        std::snprintf(label_b, sizeof(label_b), "fwd_b2 B sub=%zu", (std::size_t)sub);
        ok &= print_range_check(label_a, ((const u32*)f) + sub * sub_n * B::LANES,
                                sub_n * B::LANES, bound_mod4);
        ok &= print_range_check(label_b, ((const u32*)g) + sub * sub_n * B::LANES,
                                sub_n * B::LANES, bound_mod4);
    }
    ok &= print_range_check("forward final A", (const u32*)f, N_u32, bound_mod4);
    ok &= print_range_check("forward final B", (const u32*)g, N_u32, bound_mod4);

    S::freq_multiply(f, g, ntt_vecs);
    ok &= print_range_check("freq multiply", (const u32*)f, N_u32, bound_mod2);

    for (idt sub = 0; sub < m; ++sub) {
        S::inv_b2(f + sub * sub_n, sub_n);
        char sub_label[64];
        std::snprintf(sub_label, sizeof(sub_label), "inv_b2 sub=%zu", (std::size_t)sub);
        ok &= print_range_check(sub_label, ((const u32*)f) + sub * sub_n * B::LANES,
                                sub_n * B::LANES, bound_mod);
    }
    ok &= print_range_check("after inv_b2 full", (const u32*)f, N_u32, bound_mod);

    if (m == 3) {
        Radix3Kernel<B, Mod>::dit_pass(f, ntt_vecs, mv, roots);
    } else if (m == 5) {
        Radix5Kernel<B, Mod>::dit_pass(f, ntt_vecs, mv, roots);
    }
    if (m != 1) {
        ok &= print_range_check("outer DIT final", (const u32*)f, N_u32, bound_mod);
    }

    return ok;
}

template<u32 Mod>
static bool delta_conv_check(const std::vector<u32>& src, idt N_u32, const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using Vec = typename B::Vec;

    const idt ntt_vecs = N_u32 / B::LANES;
    ::zint::ScratchScope scope(::zint::scratch());
    Vec* f = scope.alloc<Vec>(ntt_vecs, 64);
    Vec* g = scope.alloc<Vec>(ntt_vecs, 64);
    std::vector<u32> delta(1, 1u);
    std::vector<u32> expect(N_u32, 0);
    for (std::size_t i = 0; i < src.size(); ++i) {
        expect[i] = reduce_input_modp(src[i], Mod);
    }

    ntt_conv_one_prime<B, Mod>((u32*)f, (u32*)g, ntt_vecs,
                               src.data(), (idt)src.size(),
                               delta.data(), 1, N_u32);

    const u32* out = (const u32*)f;
    for (idt i = 0; i < N_u32; ++i) {
        if (out[i] != expect[i]) {
            std::printf("  %s FAIL at u32=%zu got=%u expect=%u\n",
                        label, (std::size_t)i, out[i], expect[i]);
            return false;
        }
    }
    std::printf("  %s PASS\n", label);
    return true;
}

template<u32 Mod>
static bool small_support_conv_check(const std::vector<u32>& a32,
                                     const std::vector<u32>& b32,
                                     idt N_u32,
                                     std::size_t b_take,
                                     const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using Vec = typename B::Vec;

    std::vector<u32> b_small((std::min)(b_take, b32.size()));
    for (std::size_t i = 0; i < b_small.size(); ++i) {
        b_small[i] = b32[i];
    }

    std::vector<u32> a_red(a32.size());
    std::vector<u32> b_red(b_small.size());
    for (std::size_t i = 0; i < a32.size(); ++i) a_red[i] = reduce_input_modp(a32[i], Mod);
    for (std::size_t i = 0; i < b_small.size(); ++i) b_red[i] = reduce_input_modp(b_small[i], Mod);

    std::vector<u32> expect(N_u32, 0);
    for (std::size_t i = 0; i < a_red.size(); ++i) {
        for (std::size_t j = 0; j < b_red.size(); ++j) {
            u64 acc = expect[i + j];
            acc += (u64)a_red[i] * b_red[j];
            expect[i + j] = (u32)(acc % Mod);
        }
    }

    const idt ntt_vecs = N_u32 / B::LANES;
    ::zint::ScratchScope scope(::zint::scratch());
    Vec* f = scope.alloc<Vec>(ntt_vecs, 64);
    Vec* g = scope.alloc<Vec>(ntt_vecs, 64);

    ntt_conv_one_prime<B, Mod>((u32*)f, (u32*)g, ntt_vecs,
                               a32.data(), (idt)a32.size(),
                               b_small.data(), (idt)b_small.size(), N_u32);

    const u32* out = (const u32*)f;
    const idt check_len = (idt)std::min<std::size_t>(a_red.size() + b_red.size(), 4096);
    for (idt i = 0; i < check_len; ++i) {
        if (out[i] != expect[i]) {
            std::printf("  %s FAIL at u32=%zu got=%u expect=%u\n",
                        label, (std::size_t)i, out[i], expect[i]);
            return false;
        }
    }
    std::printf("  %s PASS\n", label);
    return true;
}

template<u32 Mod>
static u32 scalar_freq_block_out(const u32* f_raw, const u32* g_raw, u32 ww, int lane) {
    const zint::ntt::MontScalar ms{Mod};
    const u32 mod2 = ms.mod2;
    u32 ffw[8];
    u32 ff[8];
    u32 gg[8];

    for (int i = 0; i < 8; ++i) {
        gg[i] = reduce_input_modp(g_raw[i], Mod);
        u32 t = reduce_input_mod2p(f_raw[i], Mod);
        ffw[i] = ms.mul_s(t, ww);
        ff[i] = reduce_input_modp(t, Mod);
    }

    u64 acc = 0;
    for (int i = 0; i < 8; ++i) {
        const int idx = 8 - i + lane;
        const u32 term = (idx < 8) ? ffw[idx] : ff[idx - 8];
        acc += (u64)gg[i] * term;
    }

    u32 out = ms.reduce(acc);
    if (out >= mod2) out -= mod2;
    return out;
}

template<u32 Mod>
static bool freq_multiply_scalar_audit(const std::vector<u32>& a32,
                                       const std::vector<u32>& b32,
                                       idt N_u32,
                                       const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using S = NTTScheduler<B, Mod>;
    using Vec = typename B::Vec;

    const idt ntt_vecs = N_u32 / B::LANES;
    ::zint::ScratchScope scope(::zint::scratch());
    Vec* f = scope.alloc<Vec>(ntt_vecs, 64);
    Vec* g = scope.alloc<Vec>(ntt_vecs, 64);

    reduce_and_pad<B, Mod>((u32*)f, a32.data(), (idt)a32.size(), N_u32);
    reduce_and_pad<B, Mod>((u32*)g, b32.data(), (idt)b32.size(), N_u32);
    S::forward(f, ntt_vecs);
    S::forward(g, ntt_vecs);

    std::vector<u32> f_before((u32*)f, (u32*)f + N_u32);
    std::vector<u32> g_before((u32*)g, (u32*)g + N_u32);

    S::freq_multiply(f, g, ntt_vecs);
    const u32* simd_out = (const u32*)f;

    const auto& roots = S::get_roots();
    const auto& ms = S::get_ms();
    const int k = ntt_ctzll(ntt_vecs);
    const idt m = ntt_vecs >> k;
    const idt sub_n = idt{1} << k;
    const u32 omega_n = (m == 3) ? roots.tw3_root[k] : roots.tw5_root[k];
    u32 rr_init = ms.one;

    for (idt sub = 0; sub < m; ++sub) {
        u32 RR = rr_init;
        const idt base = sub * sub_n;
        for (idt i = 0; i < sub_n; i += 4) {
            const u32 RRi = ms.mul(RR, roots.img);
            const u32 ww[4] = {RR, (u32)(ms.mod2 - RR), RRi, (u32)(ms.mod2 - RRi)};

            for (int block = 0; block < 4; ++block) {
                const u32* f_raw = f_before.data() + (base + i + block) * 8;
                const u32* g_raw = g_before.data() + (base + i + block) * 8;
                const u32* got = simd_out + (base + i + block) * 8;
                for (int lane = 0; lane < 8; ++lane) {
                    const u32 expect = scalar_freq_block_out<Mod>(f_raw, g_raw, ww[block], lane);
                    if (got[lane] != expect) {
                        std::printf("  %s FAIL sub=%zu batch_vec=%zu block=%d lane=%d got=%u expect=%u ww=%u\n",
                                    label, (std::size_t)sub, (std::size_t)(base + i),
                                    block, lane, got[lane], expect, ww[block]);
                        return false;
                    }
                }
            }

            RR = ms.mul(RR, roots.RT3[ntt_ctzll(i + 4)]);
        }
        rr_init = ms.mul_s(rr_init, omega_n);
    }

    std::printf("  %s PASS\n", label);
    return true;
}

template<typename Engine>
static void force_engine_mul_u64(std::vector<u64>& out,
                                 const std::vector<u64>& a,
                                 const std::vector<u64>& b) {
    const idt na = (idt)a.size();
    const idt nb = (idt)b.size();
    const idt min_len32 = 2 * (na + nb);
    const idt N = zint::ntt::engine_ceil_size<Engine>(min_len32);
    if (N == 0) {
        std::fprintf(stderr, "engine does not fit size na=%zu nb=%zu\n", a.size(), b.size());
        std::exit(1);
    }
    std::fill(out.begin(), out.end(), 0);
    zint::ntt::big_multiply_engine<Engine>(
        (u32*)out.data(), 2 * (idt)out.size(),
        (const u32*)a.data(), 2 * na,
        (const u32*)b.data(), 2 * nb,
        N);
}

static void p50x4_mul_u64(std::vector<u64>& out,
                          const std::vector<u64>& a,
                          const std::vector<u64>& b) {
    std::fill(out.begin(), out.end(), 0);
    zint::ntt::p50x4::Ntt4& engine = zint::ntt::p50x4::Ntt4::instance();
    engine.multiply(out.data(), out.size(), a.data(), a.size(), b.data(), b.size());
}

static void run_mode(Mode mode,
                     std::vector<u64>& out,
                     const std::vector<u64>& a,
                     const std::vector<u64>& b) {
    switch (mode) {
        case Mode::current_dispatch:
            std::fill(out.begin(), out.end(), 0);
            zint::ntt::big_multiply_u64(out.data(), (idt)out.size(),
                                        a.data(), (idt)a.size(),
                                        b.data(), (idt)b.size());
            return;
        case Mode::old_dispatch:
            std::fill(out.begin(), out.end(), 0);
            old_api::big_multiply_u64(out.data(), (idt)out.size(),
                                      a.data(), (idt)a.size(),
                                      b.data(), (idt)b.size());
            return;
        case Mode::engine_a:
            force_engine_mul_u64<zint::ntt::EngineA>(out, a, b);
            return;
        case Mode::engine_b:
            force_engine_mul_u64<zint::ntt::EngineB>(out, a, b);
            return;
        case Mode::engine_c:
            force_engine_mul_u64<zint::ntt::EngineC>(out, a, b);
            return;
        case Mode::p50x4_only:
            p50x4_mul_u64(out, a, b);
            return;
    }
}

static bool find_mismatch(const std::vector<u64>& lhs,
                          const std::vector<u64>& rhs,
                          std::size_t& bad_idx) {
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            bad_idx = i;
            return true;
        }
    }
    return false;
}

static void print_u128_hex(unsigned __int128 x) {
    unsigned long long lo = (unsigned long long)x;
    unsigned long long hi = (unsigned long long)(x >> 64);
    std::printf("0x%016llx%016llx", hi, lo);
}

template<u32 Mod>
static bool truncate_b_probe(const std::vector<u32>& a32,
                             const std::vector<u32>& b32,
                             idt N_u32,
                             const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using Vec = typename B::Vec;

    const idt ntt_vecs = N_u32 / B::LANES;
    ::zint::ScratchScope scope(::zint::scratch());
    Vec* f = scope.alloc<Vec>(ntt_vecs, 64);
    Vec* g = scope.alloc<Vec>(ntt_vecs, 64);

    const u32 a0 = a32[0];
    const u32 a1 = a32[1];
    const u32 b0 = b32[0];
    const u32 b1 = b32[1];
    const u32 expect0 = (u32)((unsigned __int128)a0 * b0 % Mod);
    const u32 expect1 = (u32)(((unsigned __int128)a0 * b1 + (unsigned __int128)a1 * b0) % Mod);

    const std::size_t probes[] = {
        33, 64, 128, 256, 512, 1024, 4096, 16384, 65536,
        131072, 262144, 524288, 786432, 1048576, 1048584, 1048640,
        1048832, 1049600, 1056768, 1310656, 1310712, 1310720, 1310728, b32.size()
    };

    std::printf("  %s\n", label);
    bool ok = true;
    for (std::size_t take : probes) {
        const idt used = (idt)(std::min)(take, b32.size());
        ntt_conv_one_prime<B, Mod>((u32*)f, (u32*)g, ntt_vecs,
                                   a32.data(), (idt)a32.size(),
                                   b32.data(), used, N_u32);
        const u32 got0 = ((u32*)f)[0];
        const u32 got1 = ((u32*)f)[1];
        ok &= (got0 == expect0 && got1 == expect1);
        std::printf("    b_take=%-8zu c0=%s c1=%s got=(%u,%u)\n",
                    (std::size_t)used,
                    (got0 == expect0 ? "ok " : "BAD"),
                    (got1 == expect1 ? "ok " : "BAD"),
                    got0, got1);
    }
    return ok;
}

template<u32 Mod>
static bool sparse_high_b_probe(const std::vector<u32>& a32,
                                const std::vector<u32>& b32,
                                idt N_u32,
                                const char* label) {
    using namespace zint::ntt;
    using B = Avx2;
    using Vec = typename B::Vec;

    const idt ntt_vecs = N_u32 / B::LANES;
    ::zint::ScratchScope scope(::zint::scratch());
    Vec* f = scope.alloc<Vec>(ntt_vecs, 64);
    Vec* g = scope.alloc<Vec>(ntt_vecs, 64);

    const u32 a0 = a32[0];
    const u32 a1 = a32[1];
    const std::size_t probes[] = {1310656, 1310720, 1310728, 1310736, 1310800};

    std::printf("  %s\n", label);
    bool ok = true;
    for (std::size_t pos : probes) {
        if (pos + 1 >= b32.size()) {
            continue;
        }
        std::vector<u32> b_probe(b32.size(), 0);
        b_probe[pos] = b32[pos];
        b_probe[pos + 1] = b32[pos + 1];

        const u32 expect0 = 0;
        const u32 expect1 = 0;
        ntt_conv_one_prime<B, Mod>((u32*)f, (u32*)g, ntt_vecs,
                                   a32.data(), (idt)a32.size(),
                                   b_probe.data(), (idt)b_probe.size(), N_u32);
        const u32 got0 = ((u32*)f)[0];
        const u32 got1 = ((u32*)f)[1];
        ok &= (got0 == expect0 && got1 == expect1);
        std::printf("    spike@%-8zu c0=%s c1=%s got=(%u,%u) val=(%u,%u)\n",
                    pos,
                    (got0 == expect0 ? "ok " : "BAD"),
                    (got1 == expect1 ? "ok " : "BAD"),
                    got0, got1, b_probe[pos], b_probe[pos + 1]);
    }

    {
        std::vector<u32> b_tail(b32.size(), 0);
        for (std::size_t i = 1310720; i < b32.size(); ++i) {
            b_tail[i] = b32[i];
        }
        ntt_conv_one_prime<B, Mod>((u32*)f, (u32*)g, ntt_vecs,
                                   a32.data(), (idt)a32.size(),
                                   b_tail.data(), (idt)b_tail.size(), N_u32);
        const u32 got0 = ((u32*)f)[0];
        const u32 got1 = ((u32*)f)[1];
        ok &= (got0 == 0 && got1 == 0);
        std::printf("    tail[1310720:] c0=%s c1=%s got=(%u,%u)\n",
                    (got0 == 0 ? "ok " : "BAD"),
                    (got1 == 0 ? "ok " : "BAD"),
                    got0, got1);
    }
    return ok;
}

template<typename Engine>
static void print_engine_diag(const char* label,
                              const std::vector<u64>& a,
                              const std::vector<u64>& b) {
    using namespace zint::ntt;
    using B = Avx2;
    using Vec = typename B::Vec;

    const idt na32 = 2 * (idt)a.size();
    const idt nb32 = 2 * (idt)b.size();
    const idt min_len32 = na32 + nb32;
    const idt N = engine_ceil_size<Engine>(min_len32);
    const idt ntt_vecs = N / B::LANES;

    ::zint::ScratchScope scope(::zint::scratch());
    Vec* buf = scope.alloc<Vec>(4 * ntt_vecs, 64);
    Vec* f0 = buf + 0 * ntt_vecs;
    Vec* f1 = buf + 1 * ntt_vecs;
    Vec* f2 = buf + 2 * ntt_vecs;
    Vec* g = buf + 3 * ntt_vecs;

    ntt_conv_one_prime<B, Engine::P0>((u32*)f0, (u32*)g, ntt_vecs,
                                      (const u32*)a.data(), na32,
                                      (const u32*)b.data(), nb32, N);
    ntt_conv_one_prime<B, Engine::P1>((u32*)f1, (u32*)g, ntt_vecs,
                                      (const u32*)a.data(), na32,
                                      (const u32*)b.data(), nb32, N);
    ntt_conv_one_prime<B, Engine::P2>((u32*)f2, (u32*)g, ntt_vecs,
                                      (const u32*)a.data(), na32,
                                      (const u32*)b.data(), nb32, N);

    const u32* r0 = (const u32*)f0;
    const u32* r1 = (const u32*)f1;
    const u32* r2 = (const u32*)f2;

    const u32 a0 = (u32)(a[0] & 0xffffffffULL);
    const u32 a1 = (u32)(a[0] >> 32);
    const u32 b0 = (u32)(b[0] & 0xffffffffULL);
    const u32 b1 = (u32)(b[0] >> 32);

    const unsigned __int128 coeff0 = (unsigned __int128)a0 * b0;
    const unsigned __int128 coeff1 =
        (unsigned __int128)a0 * b1 + (unsigned __int128)a1 * b0;

    const u128 rec0 = crt_recover_e<Engine>(r0[0], r1[0], r2[0]);
    const u128 rec1 = crt_recover_e<Engine>(r0[1], r1[1], r2[1]);

    std::printf("  %s diagnostic: N=%zu min_len32=%zu\n",
                label, (std::size_t)N, (std::size_t)min_len32);

    std::printf("    coeff0 exact=");
    print_u128_hex(coeff0);
    std::printf(" recovered=0x%016llx%016llx residues=(%u,%u,%u) expect_mod=(%u,%u,%u)\n",
                (unsigned long long)rec0.hi, (unsigned long long)rec0.lo,
                r0[0], r1[0], r2[0],
                (u32)(coeff0 % Engine::P0),
                (u32)(coeff0 % Engine::P1),
                (u32)(coeff0 % Engine::P2));

    std::printf("    coeff1 exact=");
    print_u128_hex(coeff1);
    std::printf(" recovered=0x%016llx%016llx residues=(%u,%u,%u) expect_mod=(%u,%u,%u)\n",
                (unsigned long long)rec1.hi, (unsigned long long)rec1.lo,
                r0[1], r1[1], r2[1],
                (u32)(coeff1 % Engine::P0),
                (u32)(coeff1 % Engine::P1),
                (u32)(coeff1 % Engine::P2));
}

int main() {
    const std::size_t na = 1572864;
    const std::size_t nb = 786432;
    const int trials = 3;

    zint::xoshiro256pp rng(0xA5A55A5A12345678ULL);
    std::vector<u64> a(na), b(nb);
    std::vector<u64> ref(na + nb);
    std::vector<u64> out(na + nb);

    const idt min_len32 = 2 * (idt)(na + nb);
    const idt N_engine_c = zint::ntt::engine_ceil_size<zint::ntt::EngineC>(min_len32);
    const idt old_ntt_size = zint::ntt::ceil_smooth(min_len32 > 64 ? min_len32 : 64);
    bool ok = true;
    std::printf("Testing na=%zu nb=%zu min_len32=%zu\n", na, nb, (std::size_t)min_len32);
    std::printf("Current engine sizes: A=%zu B=%zu C=%zu\n",
                (std::size_t)zint::ntt::engine_ceil_size<zint::ntt::EngineA>(min_len32),
                (std::size_t)zint::ntt::engine_ceil_size<zint::ntt::EngineB>(min_len32),
                (std::size_t)zint::ntt::engine_ceil_size<zint::ntt::EngineC>(min_len32));
    std::printf("Old ceil_smooth N=%zu (old path stays on p30x3 if <= 12582912)\n",
                (std::size_t)old_ntt_size);

    if (!fill_case_from_bench_sweep(na, nb, a, b)) {
        std::fprintf(stderr, "failed to reproduce bench sweep case\n");
        return 1;
    }

    const std::vector<u32> a32 = to_u32_limbs(a);
    const std::vector<u32> b32 = to_u32_limbs(b);
    std::printf("EngineC::P0 stage checks\n");
    ok &= delta_conv_check<zint::ntt::EngineC::P0>(a32, N_engine_c, "delta convolution");
    ok &= small_support_conv_check<zint::ntt::EngineC::P0>(a32, b32, N_engine_c, 33, "small-support convolution");
    ok &= freq_multiply_scalar_audit<zint::ntt::EngineC::P0>(a32, b32, N_engine_c, "freq_multiply scalar audit");
    ok &= major_phase_range_audit<zint::ntt::EngineC::P0>(a32, b32, N_engine_c, "major-phase range audit");
    ok &= major_phase_range_audit<zint::ntt::EngineC::P1>(a32, b32, N_engine_c, "major-phase range audit (P1)");
    ok &= major_phase_range_audit<zint::ntt::EngineC::P2>(a32, b32, N_engine_c, "major-phase range audit (P2)");
    ok &= first_radix4_notw_compare<zint::ntt::EngineC::P0>(b32, (idt)b32.size(), N_engine_c, 1, "first radix4 notw compare full-b (P0, shard1)");
    ok &= first_radix4_notw_compare<zint::ntt::EngineC::P0>(b32, 1048576, N_engine_c, 1, "first radix4 notw compare trunc-b (P0, shard1)");
    ok &= truncate_b_probe<zint::ntt::EngineC::P0>(a32, b32, N_engine_c, "truncate-b probe (P0)");
    ok &= sparse_high_b_probe<zint::ntt::EngineC::P0>(a32, b32, N_engine_c, "sparse-high-b probe (P0)");

    gmp_mul_u64(ref, a, b);
    std::printf("Bench sweep case\n");
    {
        const Mode modes[] = {
            Mode::current_dispatch,
            Mode::old_dispatch,
            Mode::engine_a,
            Mode::engine_b,
            Mode::engine_c,
            Mode::p50x4_only,
        };
        for (Mode mode : modes) {
            run_mode(mode, out, a, b);
            std::size_t bad_idx = 0;
            if (find_mismatch(out, ref, bad_idx)) {
                ok = false;
                std::printf("  %-18s FAIL limb=%zu got=0x%016llx expect=0x%016llx\n",
                            mode_name(mode), bad_idx,
                            (unsigned long long)out[bad_idx],
                            (unsigned long long)ref[bad_idx]);
                if (mode == Mode::current_dispatch || mode == Mode::engine_c) {
                    print_engine_diag<zint::ntt::EngineC>("EngineC", a, b);
                }
            } else {
                std::printf("  %-18s PASS\n", mode_name(mode));
            }
        }
    }

    rng = zint::xoshiro256pp(0xA5A55A5A12345678ULL);
    for (int trial = 0; trial < trials; ++trial) {
        fill_random(a.data(), na, rng);
        fill_random(b.data(), nb, rng);
        gmp_mul_u64(ref, a, b);
        const std::vector<u32> trial_a32 = to_u32_limbs(a);
        const std::vector<u32> trial_b32 = to_u32_limbs(b);

        std::printf("Trial %d\n", trial);
        if (trial == 0) {
            ok &= major_phase_range_audit<zint::ntt::EngineC::P0>(
                trial_a32, trial_b32, N_engine_c, "major-phase range audit trial0 (P0)");
        }
        const Mode modes[] = {
            Mode::current_dispatch,
            Mode::old_dispatch,
            Mode::engine_a,
            Mode::engine_b,
            Mode::engine_c,
            Mode::p50x4_only,
        };

        for (Mode mode : modes) {
            run_mode(mode, out, a, b);
            std::size_t bad_idx = 0;
            if (find_mismatch(out, ref, bad_idx)) {
                ok = false;
                std::printf("  %-18s FAIL limb=%zu got=0x%016llx expect=0x%016llx\n",
                            mode_name(mode), bad_idx,
                            (unsigned long long)out[bad_idx],
                            (unsigned long long)ref[bad_idx]);
                if (mode == Mode::current_dispatch || mode == Mode::engine_c) {
                    print_engine_diag<zint::ntt::EngineC>("EngineC", a, b);
                }
            } else {
                std::printf("  %-18s PASS\n", mode_name(mode));
            }
        }
    }

    return ok ? 0 : 1;
}
