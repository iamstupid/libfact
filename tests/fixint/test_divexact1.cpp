#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>

#include "zfactor/fixint/uint.h"
#include "zfactor/fixint/detail/divexact1.h"

using namespace zfactor::fixint;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const char* desc) {
    if (ok) { ++g_pass; }
    else { ++g_fail; fprintf(stderr, "  FAIL: %s\n", desc); }
}

// Scalar reference: multiply dst * d and check it equals src
template<int N>
bool verify_divexact(const uint64_t* dst, const uint64_t* src, uint64_t d) {
    // Compute dst * d into product[N]
    uint64_t product[N] = {};
    uint64_t carry = 0;
    for (int i = 0; i < N; i++) {
        unsigned __int128 w = (unsigned __int128)dst[i] * d + carry;
        product[i] = (uint64_t)w;
        carry = (uint64_t)(w >> 64);
    }
    // carry should be 0 for exact fit, and product should equal src
    if (carry != 0) return false;
    return memcmp(product, src, 8 * N) == 0;
}

template<int N>
void test_divexact_case(const UInt<N>& val_in, uint64_t d, const char* tag) {
    // Ensure val * d fits in N limbs (no overflow) by clearing top bits
    UInt<N> val = val_in;
    // Divide val by d+1 to ensure val*d < 2^{64N}
    // Simpler: just clear the top ~7 bits of val so val < 2^{64N}/d
    if (d > 1) {
        unsigned shift = 64 - __builtin_clzll(d);  // ceil(log2(d+1))
        // Clear top `shift` bits of the top limb
        if (val.d[N-1] >> (64 - shift))
            val.d[N-1] &= ((uint64_t)1 << (64 - shift)) - 1;
    }

    // Compute val * d -> product (guaranteed no overflow)
    UInt<N> product{};
    uint64_t carry = 0;
    for (int i = 0; i < N; i++) {
        unsigned __int128 w = (unsigned __int128)val.d[i] * d + carry;
        product.d[i] = (uint64_t)w;
        carry = (uint64_t)(w >> 64);
    }
    // carry should be 0 now
    if (carry != 0) return;  // skip if somehow still overflows

    uint64_t inv = inverse_mod_2_64(d);
    uint64_t dst[N];
    uint64_t borrow = divexact1<N>(dst, product.d, inv, d);

    char desc[256];
    bool ok = (borrow == 0) && (memcmp(dst, val.d, 8 * N) == 0);
    if (!ok) {
        snprintf(desc, sizeof(desc), "%s: borrow=%lu dst[0]=%lx expected=%lx",
                 tag, (unsigned long)borrow, (unsigned long)dst[0], (unsigned long)val.d[0]);
    } else {
        snprintf(desc, sizeof(desc), "%s", tag);
    }
    check(ok, desc);
}

template<int N>
void test_non_exact(uint64_t d, const char* tag) {
    // A value not divisible by d: just use d+1 (odd, so d doesn't divide it unless d=1)
    UInt<N> val{};
    val.d[0] = d + 1;
    if (N >= 2) val.d[1] = 0x123456789ABCDEF0ULL;

    uint64_t inv = inverse_mod_2_64(d);
    uint64_t dst[N];
    uint64_t borrow = divexact1<N>(dst, val.d, inv, d);

    char desc[256];
    snprintf(desc, sizeof(desc), "%s: non-exact should have borrow!=0", tag);
    check(borrow != 0, desc);
}

template<int N>
void run_tests_N() {
    fprintf(stderr, "--- N = %d ---\n", N);
    std::mt19937_64 rng(42 + N);

    uint64_t test_divisors[] = {3, 5, 7, 11, 13, 17, 127, 257, 1021, 65521, 1000003,
                                 (1ULL << 31) - 1, (1ULL << 32) + 15};

    for (uint64_t d : test_divisors) {
        if (d < 3 || (d & 1) == 0) continue;  // must be odd
        char tag[128];

        // Zero
        {
            UInt<N> zero{};
            snprintf(tag, sizeof(tag), "d=%lu zero", (unsigned long)d);
            test_divexact_case<N>(zero, d, tag);
        }

        // Small values
        for (uint64_t v = 1; v <= 10; v++) {
            UInt<N> val(v);
            snprintf(tag, sizeof(tag), "d=%lu val=%lu", (unsigned long)d, (unsigned long)v);
            test_divexact_case<N>(val, d, tag);
        }

        // Random values
        for (int t = 0; t < 100; t++) {
            UInt<N> val{};
            for (int i = 0; i < N; i++) val.d[i] = rng();
            snprintf(tag, sizeof(tag), "d=%lu rand_%d", (unsigned long)d, t);
            test_divexact_case<N>(val, d, tag);
        }

        // Non-exact
        snprintf(tag, sizeof(tag), "d=%lu", (unsigned long)d);
        test_non_exact<N>(d, tag);
    }
}

int main() {
    fprintf(stderr, "=== test_divexact1 ===\n\n");

    run_tests_N<1>();
    run_tests_N<2>();
    run_tests_N<3>();
    run_tests_N<4>();
    run_tests_N<6>();
    run_tests_N<8>();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
