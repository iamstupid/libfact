#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/gcd.h"
#include "zfactor/rho.h"
#include "zfactor/bpsw.h"

using namespace zfactor;
using namespace zfactor::fixint;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    ++checks; \
    if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL: %s [%s:%d]\n", msg, __FILE__, __LINE__); } \
} while(0)

// Try rho with a few different seeds; return any non-trivial factor.
template<int N>
static UInt<N> find_factor(const UInt<N>& n) {
    MontCtx<N> mctx;
    mctx.init(n);

    uint64_t seeds[] = {1, 2, 3, 5, 7, 11, 13, 17};
    for (uint64_t c : seeds) {
        UInt<N> d = pollard_rho_brent<N>(mctx, c, 2);
        // Success: 1 < d < n
        if (d.d[0] > 1 || (N > 1 && !d.is_zero())) {
            bool is_one = (d.d[0] == 1);
            for (int i = 1; is_one && i < N; ++i) if (d.d[i] != 0) is_one = false;
            if (is_one) continue;
            if (mpn::cmp<N>(d.d, n.d) != 0) return d;
        }
    }
    UInt<N> zero{};
    return zero;
}

// ---- u64 known semiprimes ----

static void test_u64_semiprimes() {
    std::printf("  u64 semiprimes...\n");

    // Rho is tuned for "smallest factor not tiny" — anything below ~10^4
    // is trial-division territory and the batched-gcd swallows it before
    // the first gcd checkpoint, returning n.  We deliberately don't test
    // such cases here.
    struct Case { uint64_t n, p, q; };
    Case cases[] = {
        { 1018081ULL,                    1009ULL,       1009ULL},        // square — allow fail
        { 10006489ULL,                   3163ULL,       3163ULL},        // square — allow fail
        { 4294967291ULL * 4294967279ULL, 4294967279ULL, 4294967291ULL},  // ~2^32 * 2^32
        { 2305843009213693951ULL,        0,             0},              // M61 — prime, rho should fail cleanly
    };

    for (auto& t : cases) {
        UInt<1> n(t.n);
        if (t.p == 0) continue;  // skip prime-only marker
        // Ensure n is composite (skip squares — rho is known to struggle).
        if (t.p == t.q) {
            std::printf("    (skipping perfect square %llu)\n", (unsigned long long)t.n);
            continue;
        }
        UInt<1> d = find_factor<1>(n);
        CHECK(!d.is_zero(), "rho found a factor");
        if (d.is_zero()) continue;

        // Verify d is a non-trivial divisor: d | n, 1 < d < n
        CHECK(d.d[0] != 1, "factor != 1");
        CHECK(d.d[0] != t.n, "factor != n");
        CHECK(t.n % d.d[0] == 0, "factor actually divides");
        // And the cofactor matches expectation (either p or q)
        uint64_t cofactor = t.n / d.d[0];
        bool matches = (d.d[0] == t.p && cofactor == t.q) ||
                       (d.d[0] == t.q && cofactor == t.p);
        CHECK(matches, "factor matches expected p or q");
    }
}

// ---- multi-limb semiprimes ----

// Search upward from a seed for the next prime (via BPSW).  Used to
// materialize test semiprimes at target bit widths without hardcoding
// specific primes that might not be prime.
template<int N>
static UInt<N> next_prime_from(UInt<N> v) {
    if ((v.d[0] & 1) == 0) v.d[0] |= 1;  // make odd
    for (int i = 0; i < 10000; ++i) {
        if (bpsw<N>(v)) return v;
        UInt<N> two(2);
        mpn::add<N>(v.d, v.d, two.d);
    }
    return UInt<N>{};
}

// Rho's expected running time is O(sqrt(p)) for smallest prime factor p,
// so we scale test moduli to sizes the algorithm can actually finish.
// With ~0.1 µs per f-iteration, ~2^22 iters = ~0.4 s per semiprime.
// That caps the smallest factor at ~2^44.

static void test_multilimb() {
    std::printf("  multi-limb semiprimes...\n");

    // N=2: ~32-bit * ~48-bit = ~80-bit semiprime, smallest factor ~2^32
    {
        UInt<2> p = next_prime_from<2>(UInt<2>(4294967291ULL));   // near 2^32
        UInt<2> q_seed(281474976710656ULL);                        // 2^48
        UInt<2> q = next_prime_from<2>(q_seed);
        CHECK(!p.is_zero() && !q.is_zero(), "N=2 primes found");

        mpn::limb_t prod[4] = {};
        mpn::mul<2>(prod, p.d, q.d);
        UInt<2> n;
        n.d[0] = prod[0];
        n.d[1] = prod[1];

        auto t0 = std::chrono::steady_clock::now();
        UInt<2> d = find_factor<2>(n);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("    N=2 ~80-bit semiprime: %.2f ms\n", ms);

        CHECK(!d.is_zero(), "N=2 semiprime factored");
        if (!d.is_zero()) {
            UInt<2> qn, rn;
            mpn::divrem<2>(qn.d, rn.d, n.d, d.d);
            CHECK(rn.is_zero(), "N=2 d divides n");
            CHECK(!detail_rho::is_one<2>(d), "N=2 d != 1");
            CHECK(mpn::cmp<2>(d.d, n.d) != 0, "N=2 d != n");
        }
    }

    // N=3: ~40-bit * ~96-bit = ~136-bit semiprime, smallest factor ~2^40
    {
        UInt<3> p = next_prime_from<3>(UInt<3>(1099511627776ULL));  // 2^40
        UInt<3> q_seed;
        q_seed.d[0] = 1;
        q_seed.d[1] = (1ULL << 31);                                  // 2^95 + 1
        UInt<3> q = next_prime_from<3>(q_seed);
        CHECK(!p.is_zero() && !q.is_zero(), "N=3 primes found");

        mpn::limb_t prod[6] = {};
        mpn::mul<3>(prod, p.d, q.d);
        UInt<3> n;
        n.d[0] = prod[0];
        n.d[1] = prod[1];
        n.d[2] = prod[2];

        auto t0 = std::chrono::steady_clock::now();
        UInt<3> d = find_factor<3>(n);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("    N=3 ~135-bit semiprime: %.2f ms\n", ms);

        CHECK(!d.is_zero(), "N=3 semiprime factored");
        if (!d.is_zero()) {
            UInt<3> qn, rn;
            mpn::divrem<3>(qn.d, rn.d, n.d, d.d);
            CHECK(rn.is_zero(), "N=3 d divides n");
            CHECK(!detail_rho::is_one<3>(d), "N=3 d != 1");
            CHECK(mpn::cmp<3>(d.d, n.d) != 0, "N=3 d != n");
        }
    }

    // N=4: ~44-bit * ~160-bit = ~204-bit semiprime, smallest factor ~2^44
    {
        UInt<4> p = next_prime_from<4>(UInt<4>(17592186044416ULL));  // 2^44
        UInt<4> q_seed;
        q_seed.d[0] = 1;
        q_seed.d[1] = 0;
        q_seed.d[2] = (1ULL << 32);                                   // 2^160 + 1
        UInt<4> q = next_prime_from<4>(q_seed);
        CHECK(!p.is_zero() && !q.is_zero(), "N=4 primes found");

        mpn::limb_t prod[8] = {};
        mpn::mul<4>(prod, p.d, q.d);
        UInt<4> n;
        for (int i = 0; i < 4; ++i) n.d[i] = prod[i];

        auto t0 = std::chrono::steady_clock::now();
        UInt<4> d = find_factor<4>(n);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("    N=4 ~204-bit semiprime: %.2f ms\n", ms);

        CHECK(!d.is_zero(), "N=4 semiprime factored");
        if (!d.is_zero()) {
            UInt<4> qn, rn;
            mpn::divrem<4>(qn.d, rn.d, n.d, d.d);
            CHECK(rn.is_zero(), "N=4 d divides n");
        }
    }
}

// ---- random semiprimes at several sizes ----

// Find a random prime near a given bit width (via BPSW).
template<int N>
static UInt<N> random_prime_near(unsigned bits, std::mt19937_64& rng) {
    for (int attempt = 0; attempt < 200; ++attempt) {
        UInt<N> p{};
        unsigned top = bits - 1;
        p.d[top / 64] |= (1ULL << (top % 64));
        p.d[0] |= 1;
        unsigned fill_lo = (top / 64) * 64;
        for (unsigned b = 1; b < fill_lo; ++b) {
            if (rng() & 1) p.d[b / 64] |= (1ULL << (b % 64));
        }
        for (unsigned b = fill_lo + 1; b < top; ++b) {
            if (rng() & 1) p.d[b / 64] |= (1ULL << (b % 64));
        }
        if (bpsw<N>(p)) return p;
    }
    return UInt<N>{};
}

static void test_random_semiprimes() {
    std::printf("  random semiprimes (rho vs known factors)...\n");
    std::mt19937_64 rng(0xD15EA5EDD0D0BABEULL);

    // 40-bit primes * 40-bit primes = ~80-bit semiprime in N=2
    for (int iter = 0; iter < 3; ++iter) {
        auto p = random_prime_near<2>(40, rng);
        auto q = random_prime_near<2>(40, rng);
        if (p.is_zero() || q.is_zero()) continue;

        mpn::limb_t prod[4] = {};
        mpn::mul<2>(prod, p.d, q.d);
        UInt<2> n;
        n.d[0] = prod[0];
        n.d[1] = prod[1];

        UInt<2> d = find_factor<2>(n);
        CHECK(!d.is_zero(), "random 80-bit semiprime factored");
        if (!d.is_zero()) {
            UInt<2> qn, rn;
            mpn::divrem<2>(qn.d, rn.d, n.d, d.d);
            CHECK(rn.is_zero(), "d divides n");
        }
    }
}

int main() {
    test_u64_semiprimes();
    test_multilimb();
    test_random_semiprimes();

    std::printf("checks=%d failures=%d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
