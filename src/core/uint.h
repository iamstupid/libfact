#pragma once

// UInt<N> — Fixed-limb unsigned integer (N x 64-bit, little-endian).
// sizeof(UInt<N>) == 8*N, zero-initialized by default.

#include <cstdint>

namespace zfactor {

template<int N>
struct UInt {
    static_assert(N >= 1 && N <= 16, "N must be 1..16");

    uint64_t d[N] = {};
    static constexpr int limbs = N;

    uint64_t& operator[](int i) { return d[i]; }
    const uint64_t& operator[](int i) const { return d[i]; }
};

} // namespace zfactor
