#pragma once

#include <cstdint>
#include <intrin.h>

namespace zfactor::mpn {

using limb_t = std::uint64_t;

inline limb_t umul128(limb_t a, limb_t b, limb_t* hi) {
    return _umul128(a, b, reinterpret_cast<unsigned __int64*>(hi));
}

inline uint8_t addcarry(uint8_t c_in, limb_t a, limb_t b, limb_t* r) {
    return _addcarry_u64(c_in, a, b, reinterpret_cast<unsigned __int64*>(r));
}

inline uint8_t subborrow(uint8_t b_in, limb_t a, limb_t b, limb_t* r) {
    return _subborrow_u64(b_in, a, b, reinterpret_cast<unsigned __int64*>(r));
}

template<int N>
inline limb_t add(limb_t* r, const limb_t* a, const limb_t* b) {
    limb_t cy = 0;
    for (int i = 0; i < N; ++i)
        cy = addcarry((uint8_t)cy, a[i], b[i], r + i);
    return cy;
}

template<int N>
inline limb_t sub(limb_t* r, const limb_t* a, const limb_t* b) {
    limb_t bw = 0;
    for (int i = 0; i < N; ++i)
        bw = subborrow((uint8_t)bw, a[i], b[i], r + i);
    return bw;
}

template<int N>
inline limb_t addmul1(limb_t* r, const limb_t* a, limb_t b) {
    limb_t cy = 0;
    for (int i = 0; i < N; ++i) {
#if defined(ZFACTOR_HAS_BMI2)
        limb_t hi = 0;
        limb_t lo = _mulx_u64(a[i], b, reinterpret_cast<unsigned __int64*>(&hi));
#else
        limb_t hi = 0;
        limb_t lo = _umul128(a[i], b, reinterpret_cast<unsigned __int64*>(&hi));
#endif
        limb_t sum = 0;
        uint8_t c1 = addcarry(0, lo, r[i], &sum);
        uint8_t c2 = addcarry(0, sum, cy, &sum);
        r[i] = sum;
        cy = hi + c1 + c2;
    }
    return cy;
}

} // namespace zfactor::mpn

#include "mpn_common.h"
