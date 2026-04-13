// Minimal debug: verify new engines against hand-computed small cases.
#include <cstdio>
#include <cstdint>
#include <vector>
#include "zint/zint.hpp"
#include "zint/ntt/api.hpp"

using u32 = uint32_t;

int main() {
    // Test 1: multiply two single-limb values, 1 * 1 = 1.
    {
        std::vector<u32> a = {1, 0, 0, 0};
        std::vector<u32> b = {1, 0, 0, 0};
        std::vector<u32> out(8, 0xDEADBEEF);
        bool ok = zint::ntt::big_multiply(out.data(), out.size(), a.data(), 4, b.data(), 4);
        std::printf("Test 1: 1*1. ok=%d out=", ok);
        for (auto v : out) std::printf("%u ", v);
        std::printf("(expect 1 0 0 0 0 0 0 0)\n");
    }

    // Test 2: multiply two specific values whose product we can compute.
    // a = 2, b = 3, expect 6.
    {
        std::vector<u32> a = {2, 0, 0, 0};
        std::vector<u32> b = {3, 0, 0, 0};
        std::vector<u32> out(8, 0xDEADBEEF);
        bool ok = zint::ntt::big_multiply(out.data(), out.size(), a.data(), 4, b.data(), 4);
        std::printf("Test 2: 2*3. ok=%d out=", ok);
        for (auto v : out) std::printf("%u ", v);
        std::printf("(expect 6 0 0 0 0 0 0 0)\n");
    }

    // Test 3: near-max u32 values: 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE00000001
    {
        std::vector<u32> a = {0xFFFFFFFFu, 0, 0, 0};
        std::vector<u32> b = {0xFFFFFFFFu, 0, 0, 0};
        std::vector<u32> out(8, 0xDEADBEEF);
        bool ok = zint::ntt::big_multiply(out.data(), out.size(), a.data(), 4, b.data(), 4);
        std::printf("Test 3: (2^32-1)^2. ok=%d out=", ok);
        for (auto v : out) std::printf("%u ", v);
        std::printf("(expect 1 4294967294 0 0 0 0 0 0)\n");
    }

    // Test 4: 2^128 - 1 squared.
    {
        std::vector<u32> a = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
        std::vector<u32> b = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
        std::vector<u32> out(8, 0xDEADBEEF);
        bool ok = zint::ntt::big_multiply(out.data(), out.size(), a.data(), 4, b.data(), 4);
        std::printf("Test 4: (2^128-1)^2. ok=%d out=", ok);
        for (auto v : out) std::printf("%u ", v);
        // (2^128-1)^2 = 2^256 - 2^129 + 1 =  ...FFFE 0000... 0001
        std::printf("(expect 1 0 0 0 FFFFFFFE FFFFFFFF FFFFFFFF FFFFFFFF)\n");
    }

    return 0;
}
