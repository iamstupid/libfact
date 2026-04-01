#pragma once

// Cross-platform intrinsics: mul128, addcarry, subborrow.
// MSVC uses <intrin.h>, GCC/Clang use __int128.

#include <cstdint>
#include <utility>

#if defined(_MSC_VER)
    #include <intrin.h>
    #define ZFACTOR_MSVC 1
#elif defined(__GNUC__) || defined(__clang__)
    #define ZFACTOR_GCC_COMPAT 1
    using u128 = unsigned __int128;
#endif

namespace zfactor {

inline std::pair<uint64_t, uint64_t> mul128(uint64_t a, uint64_t b) {
#if ZFACTOR_MSVC
    uint64_t hi;
    uint64_t lo = _umul128(a, b, &hi);
    return {lo, hi};
#else
    u128 r = static_cast<u128>(a) * b;
    return {static_cast<uint64_t>(r), static_cast<uint64_t>(r >> 64)};
#endif
}

inline uint64_t mulhi(uint64_t a, uint64_t b) {
#if ZFACTOR_MSVC
    uint64_t hi;
    _umul128(a, b, &hi);
    return hi;
#else
    return static_cast<uint64_t>((static_cast<u128>(a) * b) >> 64);
#endif
}

inline uint8_t addcarry(uint8_t c, uint64_t a, uint64_t b, uint64_t* r) {
#if ZFACTOR_MSVC
    return _addcarry_u64(c, a, b, reinterpret_cast<unsigned long long*>(r));
#else
    u128 sum = static_cast<u128>(a) + b + c;
    *r = static_cast<uint64_t>(sum);
    return static_cast<uint8_t>(sum >> 64);
#endif
}

inline uint8_t subborrow(uint8_t b_in, uint64_t a, uint64_t b, uint64_t* r) {
#if ZFACTOR_MSVC
    return _subborrow_u64(b_in, a, b, reinterpret_cast<unsigned long long*>(r));
#else
    u128 diff = static_cast<u128>(a) - b - b_in;
    *r = static_cast<uint64_t>(diff);
    return static_cast<uint8_t>(diff >> 64) & 1;
#endif
}

} // namespace zfactor
