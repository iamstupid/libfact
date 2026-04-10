#include <cstdio>
#include <cstdlib>
#include <random>
#include "zfactor/fixint/modular.h"

using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

template<int N>
UInt<N> random_mod(std::mt19937_64& rng, const UInt<N>& mod) {
    UInt<N> r;
    for (int i = 0; i < N; ++i) r.d[i] = rng();
    unsigned mod_bits = mpn::bit_length<N>(mod.d);
    if (mod_bits < unsigned(N) * 64) {
        unsigned top_limb = mod_bits / 64;
        unsigned top_bit = mod_bits % 64;
        for (unsigned i = top_limb + 1; i < unsigned(N); ++i) r.d[i] = 0;
        if (top_limb < unsigned(N) && top_bit < 64)
            r.d[top_limb] &= (1ULL << (top_bit + 1)) - 1;
    }
    while (mpn::cmp<N>(r.d, mod.d) >= 0)
        mpn::sub<N>(r.d, r.d, mod.d);
    return r;
}

// Reference mulmod using 2N+1 limb buffer for shifts
template<int N>
UInt<N> ref_mulmod(const UInt<N>& a, const UInt<N>& b, const UInt<N>& n) {
    constexpr int W = 2 * N + 1;  // wide buffer
    mpn::limb_t prod[W] = {};
    mpn::mul<N>(prod, a.d, b.d);
    // prod has at most 2N limbs of real data; prod[2N] = 0
    unsigned n_bits = mpn::bit_length<N>(n.d);
    unsigned prod_bits = 0;
    for (int i = W - 1; i >= 0; --i)
        if (prod[i]) { prod_bits = unsigned(i)*64 + 64 - unsigned(__builtin_clzll(prod[i])); break; }
    if (prod_bits == 0 || n_bits == 0) { UInt<N> r; return r; }
    for (int s = int(prod_bits) - int(n_bits); s >= 0; --s) {
        mpn::limb_t sn[W] = {};
        unsigned ls = unsigned(s)/64, bs = unsigned(s)%64;
        for (unsigned i = 0; i < unsigned(N); ++i) {
            unsigned d = i + ls;
            if (d < unsigned(W)) {
                sn[d] |= n.d[i] << bs;
                if (bs && d+1 < unsigned(W))
                    sn[d+1] |= n.d[i] >> (64-bs);
            }
        }
        bool ge = true;
        for (int i = W-1; i >= 0; --i) {
            if (prod[i] < sn[i]) { ge=false; break; }
            if (prod[i] > sn[i]) break;
        }
        if (ge) {
            uint8_t bw = 0;
            for (int i = 0; i < W; ++i) {
                unsigned __int128 s128 = (unsigned __int128)prod[i] - sn[i] - bw;
                prod[i] = (mpn::limb_t)s128;
                bw = (uint8_t)((mpn::limb_t)(s128>>64)&1);
            }
        }
    }
    UInt<N> r; for (int i = 0; i < N; ++i) r.d[i] = prod[i]; return r;
}

template<int N>
mpn::limb_t ref_submul1(mpn::limb_t* r, const mpn::limb_t* a, mpn::limb_t b) {
    mpn::limb_t bw = 0;
    for (int i = 0; i < N; ++i) {
        unsigned __int128 prod = (unsigned __int128)a[i] * b + bw;
        mpn::limb_t lo = (mpn::limb_t)prod, hi = (mpn::limb_t)(prod>>64);
        bw = hi + (r[i] < lo); r[i] -= lo;
    }
    return bw;
}

template<int N>
void test_all() {
    std::mt19937_64 rng(42 + N);
    std::fprintf(stderr, "  Testing N=%d...\n", N);

    // submul1
    for (int t = 0; t < 100; ++t) {
        mpn::limb_t r1[N], r2[N], a[N];
        for (int i = 0; i < N; ++i) { r1[i] = r2[i] = rng(); a[i] = rng(); }
        mpn::limb_t scalar = rng();
        auto bw1 = mpn::submul1<N>(r1, a, scalar);
        auto bw2 = ref_submul1<N>(r2, a, scalar);
        bool ok = (bw1 == bw2);
        for (int i = 0; i < N; ++i) ok &= (r1[i] == r2[i]);
        CHECK(ok, "submul1");
    }

    // MontCtx
    UInt<N> n; for (int i = 0; i < N; ++i) n.d[i] = rng();
    n.d[0] |= 1; if (N > 1) n.d[N-1] |= (1ULL << 63);
    MontCtx<N> ctx; ctx.init(n);

    CHECK(n.d[0] * ctx.pos_inv == 1, "pos_inv");
    CHECK(mpn::cmp<N>(ctx.r_mod.d, ctx.mod.d) < 0, "r_mod < mod");
    CHECK(mpn::cmp<N>(ctx.r2_mod.d, ctx.mod.d) < 0, "r2_mod < mod");
    CHECK(!mpn::is_zero<N>(ctx.r_mod.d), "r_mod != 0");

    { MontScope<N> scope(ctx);

    // Round-trip
    for (int t = 0; t < 30; ++t) {
        UInt<N> a = random_mod<N>(rng, n);
        auto ma = Mod<N>::from_uint(a);
        CHECK(ma.to_uint() == a, "round-trip");
    }

    CHECK(Mod<N>::from_uint(UInt<N>(1)) == Mod<N>::one(), "mont one");
    CHECK(Mod<N>::from_uint(UInt<N>(1)).to_uint() == UInt<N>(1), "mont one val");

    // Montmul
    for (int t = 0; t < 30; ++t) {
        UInt<N> a = random_mod<N>(rng, n), b = random_mod<N>(rng, n);
        auto ma = Mod<N>::from_uint(a), mb = Mod<N>::from_uint(b);
        auto result = (ma * mb).to_uint();
        auto expected = ref_mulmod<N>(a, b, n);
        CHECK(result == expected, "montmul");
    }

    // Mul by 0
    { auto a = random_mod<N>(rng, n); auto ma = Mod<N>::from_uint(a);
      CHECK((ma * Mod<N>::from_uint(UInt<N>(0))).to_uint().is_zero(), "mul by 0"); }

    // Montsqr
    for (int t = 0; t < 30; ++t) {
        UInt<N> a = random_mod<N>(rng, n);
        auto ma = Mod<N>::from_uint(a);
        CHECK(ma.sqr() == ma * ma, "sqr");
    }

    // Add/sub
    for (int t = 0; t < 30; ++t) {
        UInt<N> a = random_mod<N>(rng, n), b = random_mod<N>(rng, n);
        auto ma = Mod<N>::from_uint(a), mb = Mod<N>::from_uint(b);
        CHECK((ma + mb) - mb == ma, "add-sub");
        CHECK((ma - mb) + mb == ma, "sub-add");
    }

    // fmadd/fmsub
    for (int t = 0; t < 20; ++t) {
        UInt<N> a = random_mod<N>(rng, n), b = random_mod<N>(rng, n), c = random_mod<N>(rng, n);
        auto ma = Mod<N>::from_uint(a), mb = Mod<N>::from_uint(b), mc = Mod<N>::from_uint(c);
        CHECK(fmadd<N>(ma, mb, mc) == ma*mb + mc, "fmadd");
        CHECK(fmsub<N>(ma, mb, mc) == ma*mb - mc, "fmsub");
    }

    // ModWide
    for (int t = 0; t < 20; ++t) {
        UInt<N> a = random_mod<N>(rng, n), b = random_mod<N>(rng, n);
        auto ma = Mod<N>::from_uint(a), mb = Mod<N>::from_uint(b);
        CHECK(mul_wide<N>(ma, mb).redc() == ma * mb, "wide redc");
    }

    } // scope

    // Fermat p=97
    { UInt<N> p(97); MontCtx<N> c97; c97.init(p); MontScope<N> s97(c97);
      for (int v = 1; v <= 96; v += 7) {
          CHECK(pow<N>(Mod<N>::from_uint(UInt<N>(uint64_t(v))), UInt<N>(96)) == Mod<N>::one(), "fermat97");
      } }

    // Fermat Mersenne 2^61-1
    { UInt<N> p(2305843009213693951ULL); MontCtx<N> cp; cp.init(p); MontScope<N> sp(cp);
      CHECK(pow<N>(Mod<N>::from_uint(UInt<N>(2)), UInt<N>(2305843009213693950ULL)) == Mod<N>::one(), "fermat_mersenne"); }

    // Edge: mod=3
    { UInt<N> p(3); MontCtx<N> c3; c3.init(p); MontScope<N> s3(c3);
      auto m1 = Mod<N>::from_uint(UInt<N>(1)), m2 = Mod<N>::from_uint(UInt<N>(2));
      CHECK((m1 + m2).to_uint().is_zero(), "1+2=0 mod3");
      CHECK((m2 * m2) == m1, "2*2=1 mod3"); }
}

int main() {
    std::fprintf(stderr, "Montgomery test suite\n");
    test_all<1>();
    test_all<2>();
    test_all<3>();
    test_all<4>();
    test_all<5>();
    test_all<6>();
    test_all<8>();
    std::fprintf(stderr, "\n%d checks, %d failures\n", checks, failures);
    return failures > 0 ? 1 : 0;
}
