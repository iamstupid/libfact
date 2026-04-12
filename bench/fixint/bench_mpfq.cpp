// Benchmark: libfact montmul  vs  MPFQ montmul  vs  MPFQ montsqr.
//
// MPFQ kernels are taken verbatim from the Perl-generated x86_64 header
// (kwantam/mpfq, gen_mp_x86_64.pl).  We follow MPFQ's elt.pm conventions:
//
//   mul(z, x, y) = mul_N(tmp, x, y); redc_N(z, tmp, inv, p);
//   sqr(z, x)    = sqr_N(tmp, x);    redc_N(z, tmp, inv, p);
//
// where the size N=1..4 path uses MPFQ's hand-tuned assembly for both
// mul_N and sqr_N.  MPFQ's redc_N is the C wrapper from gen_mp_longlong.pl
// reproduced inline below — it calls the asm addmul1_N kernels.
//
// All benches are latency-bound: each iteration consumes the previous
// iteration's output via a[k+1] = r[k], matching how Mont ops are chained
// inside ECM/EECM/rho inner loops.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

// MPFQ kernels expect mp_limb_t to exist and the C-style declarations.
typedef uint64_t mp_limb_t;
#define MAYBE_UNUSED __attribute__((unused))

// MPFQ's mul_1 / sqr_1 use the GMP umul_ppmm macro (64x64 → 128).
// We provide it via __int128 — same codegen on x86_64.
#define umul_ppmm(hi, lo, a, b)                                  \
    do {                                                         \
        unsigned __int128 __p = (unsigned __int128)(a) * (b);    \
        (lo) = (uint64_t)__p;                                    \
        (hi) = (uint64_t)(__p >> 64);                            \
    } while (0)

// Pull in MPFQ's hand-tuned x86_64 kernels (mul_1..9, sqr_1..4, addmul1_1..9,
// addmul1_nc_1..9, add_nc_1..9, sub_nc_1..9, mulredc_1).  All static inline.
#include "mpfq_kernels_x86_64.h"

// libfact mont layer.
#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/montgomery.h"

// ============================================================================
// MPFQ-style redc_N reproduced from gen_mp_longlong.pl's redc_k template.
// Uses the asm addmul1_N kernels above.  We need a few helpers (add with
// carry, sub with carry, n-limb compare) that MPFQ takes from the longlong
// fallback header — we inline equivalents here so we can stay in one TU.
// ============================================================================

static inline uint64_t add_n_with_cy(uint64_t* z, const uint64_t* x,
                                     const uint64_t* y, int n) {
    uint64_t cy = 0;
    for (int i = 0; i < n; ++i) {
        uint64_t a = x[i], b = y[i];
        uint64_t s = a + b;
        uint64_t c1 = (s < a);
        uint64_t s2 = s + cy;
        uint64_t c2 = (s2 < s);
        z[i] = s2;
        cy = c1 + c2;
    }
    return cy;
}

static inline uint64_t sub_n_with_bw(uint64_t* z, const uint64_t* x,
                                     const uint64_t* y, int n) {
    uint64_t bw = 0;
    for (int i = 0; i < n; ++i) {
        uint64_t a = x[i], b = y[i];
        uint64_t d = a - b;
        uint64_t b1 = (a < b);
        uint64_t d2 = d - bw;
        uint64_t b2 = (d < bw);
        z[i] = d2;
        bw = b1 + b2;
    }
    return bw;
}

static inline int cmp_n(const uint64_t* x, const uint64_t* y, int n) {
    for (int i = n - 1; i >= 0; --i) {
        if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    }
    return 0;
}

// Bridge: dispatch to MPFQ's asm addmul1_N at compile time.
template<int N>
[[gnu::always_inline]] static inline uint64_t mpfq_addmul1(uint64_t* z, const uint64_t* x, uint64_t c) {
    if constexpr (N == 1) return addmul1_1(z, x, c);
    else if constexpr (N == 2) return addmul1_2(z, x, c);
    else if constexpr (N == 3) return addmul1_3(z, x, c);
    else if constexpr (N == 4) return addmul1_4(z, x, c);
    else return 0;
}

template<int N>
[[gnu::always_inline]] static inline void mpfq_mul_kernel(uint64_t* tmp, const uint64_t* x, const uint64_t* y) {
    if constexpr (N == 1) mul_1(tmp, x, y);
    else if constexpr (N == 2) mul_2(tmp, x, y);
    else if constexpr (N == 3) mul_3(tmp, x, y);
    else if constexpr (N == 4) mul_4(tmp, x, y);
}

template<int N>
[[gnu::always_inline]] static inline void mpfq_sqr_kernel(uint64_t* tmp, const uint64_t* x) {
    if constexpr (N == 1) sqr_1(tmp, x);
    else if constexpr (N == 2) sqr_2(tmp, x);
    else if constexpr (N == 3) sqr_3(tmp, x);
    else if constexpr (N == 4) sqr_4(tmp, x);
}

// Replicate MPFQ's redc_k template (from gen_mp_longlong.pl line 1200).
//
//   for (i = 0; i < N; ++i) {
//       t = x[i] * inv;
//       cy = addmul1_N(x+i, p, t);
//       x[i] = cy;            // x[i] was zeroed by the addmul; stash carry
//   }
//   cy = add_{N-1}(x+N+1, x+N+1, x);   // fold low-half carries into high half
//   cy += x[N-1];
//   if (cy || cmp(x+N, p) >= 0) sub_N(z, x+N, p);
//   else                       copy_N(z, x+N);
//
template<int N>
[[gnu::always_inline]] static inline void mpfq_redc(uint64_t* z, uint64_t* x,
                                                    uint64_t inv, const uint64_t* p) {
    uint64_t cy = 0;
    for (int i = 0; i < N; ++i) {
        uint64_t t = x[i] * inv;
        cy = mpfq_addmul1<N>(x + i, p, t);
        x[i] = cy;
    }
    if constexpr (N >= 2) {
        cy = add_n_with_cy(x + N + 1, x + N + 1, x, N - 1);
    } else {
        cy = 0;
    }
    cy += x[N - 1];
    if (cy || cmp_n(x + N, p, N) >= 0) {
        sub_n_with_bw(z, x + N, p, N);
    } else {
        for (int i = 0; i < N; ++i) z[i] = x[i + N];
    }
}

template<int N>
[[gnu::always_inline]] static inline void mpfq_mul(uint64_t* z, const uint64_t* x, const uint64_t* y,
                                                   uint64_t inv, const uint64_t* p) {
    uint64_t tmp[2 * N];
    mpfq_mul_kernel<N>(tmp, x, y);
    mpfq_redc<N>(z, tmp, inv, p);
}

template<int N>
[[gnu::always_inline]] static inline void mpfq_sqr(uint64_t* z, const uint64_t* x,
                                                   uint64_t inv, const uint64_t* p) {
    uint64_t tmp[2 * N];
    mpfq_sqr_kernel<N>(tmp, x);
    mpfq_redc<N>(z, tmp, inv, p);
}

// ============================================================================
// Bench drivers — all latency-bound (each iter feeds the prior iter's output).
// ============================================================================

template<int N>
static void make_modulus(uint64_t* p) {
    std::mt19937_64 rng(0xC0FFEE0DDBEEFBEAULL + N);
    for (int i = 0; i < N; ++i) p[i] = rng();
    p[0] |= 1ull;                        // odd
    p[N - 1] |= (1ull << 63);            // full width
}

// asm volatile barrier — forces the compiler to treat `v` as opaque,
// blocking constant folding across loop iterations.  Needed for libfact's
// pure-C montmul_1/2 paths, which the compiler would otherwise CSE through.
template<typename T>
[[gnu::always_inline]] static inline void escape(T& v) {
    asm volatile("" : "+r,m"(v) : : "memory");
}

template<int N>
[[gnu::noinline]]
static uint64_t bench_libfact_mul(const zfactor::fixint::MontCtx<N>& mctx, uint64_t iters) {
    using namespace zfactor::fixint;
    UInt<N> a{}, b{}, r{};
    for (int i = 0; i < N; ++i) {
        a.d[i] = (i + 1) * 0x9E3779B97F4A7C15ULL;
        b.d[i] = (i + 7) * 0x517CC1B727220A95ULL;
    }
    a.d[N - 1] &= 0x7FFFFFFFFFFFFFFFULL;
    b.d[N - 1] &= 0x7FFFFFFFFFFFFFFFULL;
    for (uint64_t i = 0; i < iters; ++i) {
        montmul<N>(r.d, a.d, b.d, mctx);
        for (int k = 0; k < N; ++k) a.d[k] = r.d[k];
        for (int k = 0; k < N; ++k) escape(a.d[k]);
    }
    uint64_t sink = 0;
    for (int k = 0; k < N; ++k) sink ^= r.d[k];
    return sink;
}

// Inline-asm variant — only meaningful at N=1 (the asm bypass).
[[gnu::noinline]]
static uint64_t bench_libfact_mul_1_asm(const zfactor::fixint::MontCtx<1>& mctx, uint64_t iters) {
    using namespace zfactor::fixint;
    UInt<1> a{}, b{}, r{};
    a.d[0] = 0x9E3779B97F4A7C15ULL;
    b.d[0] = 0x517CC1B727220A95ULL & 0x7FFFFFFFFFFFFFFFULL;
    for (uint64_t i = 0; i < iters; ++i) {
        montmul_1_asm(r.d, a.d, b.d, mctx.mod.d[0], mctx.pos_inv);
        a.d[0] = r.d[0];
        escape(a.d[0]);
    }
    return r.d[0];
}

template<int N>
[[gnu::noinline]]
static uint64_t bench_libfact_sqr(const zfactor::fixint::MontCtx<N>& mctx, uint64_t iters) {
    using namespace zfactor::fixint;
    UInt<N> a{}, r{};
    for (int i = 0; i < N; ++i) a.d[i] = (i + 1) * 0x9E3779B97F4A7C15ULL;
    a.d[N - 1] &= 0x7FFFFFFFFFFFFFFFULL;
    for (uint64_t i = 0; i < iters; ++i) {
        montsqr<N>(r.d, a.d, mctx);
        for (int k = 0; k < N; ++k) a.d[k] = r.d[k];
        for (int k = 0; k < N; ++k) escape(a.d[k]);
    }
    uint64_t sink = 0;
    for (int k = 0; k < N; ++k) sink ^= r.d[k];
    return sink;
}

template<int N>
[[gnu::noinline]]
static uint64_t bench_mpfq_mul(const uint64_t* p, uint64_t inv, uint64_t iters) {
    uint64_t a[N], b[N], r[N];
    for (int i = 0; i < N; ++i) {
        a[i] = (i + 1) * 0x9E3779B97F4A7C15ULL;
        b[i] = (i + 7) * 0x517CC1B727220A95ULL;
    }
    a[N - 1] &= 0x7FFFFFFFFFFFFFFFULL;
    b[N - 1] &= 0x7FFFFFFFFFFFFFFFULL;
    for (uint64_t i = 0; i < iters; ++i) {
        mpfq_mul<N>(r, a, b, inv, p);
        for (int k = 0; k < N; ++k) a[k] = r[k];
        for (int k = 0; k < N; ++k) escape(a[k]);
    }
    uint64_t sink = 0;
    for (int k = 0; k < N; ++k) sink ^= r[k];
    return sink;
}

template<int N>
[[gnu::noinline]]
static uint64_t bench_mpfq_sqr(const uint64_t* p, uint64_t inv, uint64_t iters) {
    uint64_t a[N], r[N];
    for (int i = 0; i < N; ++i) a[i] = (i + 1) * 0x9E3779B97F4A7C15ULL;
    a[N - 1] &= 0x7FFFFFFFFFFFFFFFULL;
    for (uint64_t i = 0; i < iters; ++i) {
        mpfq_sqr<N>(r, a, inv, p);
        for (int k = 0; k < N; ++k) a[k] = r[k];
        for (int k = 0; k < N; ++k) escape(a[k]);
    }
    uint64_t sink = 0;
    for (int k = 0; k < N; ++k) sink ^= r[k];
    return sink;
}

template<int N>
static void run_one() {
    using namespace zfactor::fixint;
    UInt<N> n_uint{};
    uint64_t p[N];
    make_modulus<N>(p);
    for (int i = 0; i < N; ++i) n_uint.d[i] = p[i];

    MontCtx<N> mctx;
    mctx.init(n_uint);
    uint64_t inv = mctx.neg_inv;   // MPFQ uses negative inverse mod 2^64

    constexpr uint64_t WARMUP = 1 << 16;
    (void)bench_libfact_mul<N>(mctx, WARMUP);
    (void)bench_libfact_sqr<N>(mctx, WARMUP);
    (void)bench_mpfq_mul<N>(p, inv, WARMUP);
    (void)bench_mpfq_sqr<N>(p, inv, WARMUP);

    constexpr uint64_t ITERS = 50'000'000;
    auto t0 = std::chrono::steady_clock::now();
    uint64_t s_lm = bench_libfact_mul<N>(mctx, ITERS);
    auto t1 = std::chrono::steady_clock::now();
    uint64_t s_ls = bench_libfact_sqr<N>(mctx, ITERS);
    auto t2 = std::chrono::steady_clock::now();
    uint64_t s_mm = bench_mpfq_mul<N>(p, inv, ITERS);
    auto t3 = std::chrono::steady_clock::now();
    uint64_t s_ms = bench_mpfq_sqr<N>(p, inv, ITERS);
    auto t4 = std::chrono::steady_clock::now();

    auto ns = [&](auto a, auto b) {
        return std::chrono::duration<double, std::nano>(b - a).count() / ITERS;
    };
    double libfact_mul_ns = ns(t0, t1);
    double libfact_sqr_ns = ns(t1, t2);
    double mpfq_mul_ns    = ns(t2, t3);
    double mpfq_sqr_ns    = ns(t3, t4);

    if constexpr (N == 1) {
        // Extra column: hand-written inline asm variant.
        (void)bench_libfact_mul_1_asm(mctx, WARMUP);
        auto t5 = std::chrono::steady_clock::now();
        uint64_t s_la = bench_libfact_mul_1_asm(mctx, ITERS);
        auto t6 = std::chrono::steady_clock::now();
        double libfact_asm_ns = ns(t5, t6);
        std::printf("  N=%d   lib_mul=%6.2f  lib_asm=%6.2f  lib_sqr=%6.2f   "
                    "mpfq_mul=%6.2f  mpfq_sqr=%6.2f   "
                    "(asm/c++=%.2fx)\n",
                    N, libfact_mul_ns, libfact_asm_ns, libfact_sqr_ns,
                    mpfq_mul_ns, mpfq_sqr_ns,
                    libfact_asm_ns / libfact_mul_ns);
        (void)s_la;
    } else {
        std::printf("  N=%d   lib_mul=%6.2f  lib_sqr=%6.2f   "
                    "mpfq_mul=%6.2f  mpfq_sqr=%6.2f   "
                    "(lib/mpfq=%.2fx)\n",
                    N, libfact_mul_ns, libfact_sqr_ns,
                    mpfq_mul_ns, mpfq_sqr_ns,
                    libfact_mul_ns / mpfq_mul_ns);
    }
    (void)s_lm; (void)s_ls; (void)s_mm; (void)s_ms;
}

int main() {
    std::printf("Mont op latency: libfact (Mont CIOS / __int128) vs MPFQ (mul_N + redc_N)\n");
    std::printf("All numbers are ns/iter, latency-bound (output feeds next input).\n\n");
    run_one<1>();
    run_one<2>();
    run_one<3>();
    run_one<4>();
    return 0;
}
