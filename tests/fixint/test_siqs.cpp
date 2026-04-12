#include <cstdio>
#include <cstdint>
#include <cassert>

#include "zfactor/fixint/uint.h"
#include "zfactor/siqs.h"

using namespace zfactor;
using namespace zfactor::fixint;

template<int N>
UInt<N> from_u64(uint64_t v) {
    UInt<N> r = {};
    r.d[0] = v;
    return r;
}

template<int N>
UInt<N> mul_u64(const UInt<N>& a, uint64_t b) {
    UInt<N> r = {};
    uint64_t carry = 0;
    for (int i = 0; i < N; i++) {
        unsigned __int128 w = (unsigned __int128)a.d[i] * b + carry;
        r.d[i] = (uint64_t)w;
        carry = (uint64_t)(w >> 64);
    }
    return r;
}

template<int N>
bool is_one(const UInt<N>& a) {
    if (a.d[0] != 1) return false;
    for (int i = 1; i < N; i++) if (a.d[i] != 0) return false;
    return true;
}

// Test factoring a product of two primes
template<int N>
bool test_factor(uint64_t p, uint64_t q, const char* label) {
    UInt<N> n = mul_u64(from_u64<N>(p), q);
    printf("Testing %s: %lu * %lu (%u bits)... ", label, p, q, n.bit_length());
    fflush(stdout);

    auto factor = siqs::siqs<N>(n);
    if (!factor) {
        printf("FAILED (no factor found)\n");
        return false;
    }

    // Verify factor divides n
    UInt<N> qr, rem;
    mpn::divrem<N>(qr.d, rem.d, n.d, factor->d);
    if (!rem.is_zero()) {
        printf("FAILED (factor doesn't divide n)\n");
        return false;
    }
    // Verify non-trivial
    if (is_one(*factor) || mpn::cmp<N>(factor->d, n.d) == 0) {
        printf("FAILED (trivial factor)\n");
        return false;
    }

    printf("OK (found factor)\n");
    return true;
}

int main() {
    printf("=== SIQS Test Suite ===\n\n");

    int pass = 0, fail = 0;

    // 80-bit semiprimes (N=2 sufficient)
    if (test_factor<2>(1000000007ULL, 1000000009ULL, "80-bit")) pass++; else fail++;
    if (test_factor<2>(2147483647ULL, 2147483629ULL, "~62-bit")) pass++; else fail++;

    // 96-bit semiprime
    if (test_factor<2>(4294967291ULL, 4294967279ULL, "96-bit")) pass++; else fail++;

    // 100-bit: needs slightly larger primes
    if (test_factor<2>(1099511627689ULL, 1099511627563ULL, "~100-bit")) pass++; else fail++;

    printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);
    return fail ? 1 : 0;
}
