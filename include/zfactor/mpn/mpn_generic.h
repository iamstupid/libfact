#pragma once

#include <cstdint>
#include <utility>

namespace zfactor::mpn {

using limb_t = std::uint64_t;

inline limb_t umul128(limb_t a, limb_t b, limb_t* hi) {
    unsigned __int128 p = (unsigned __int128)a * b;
    *hi = (limb_t)(p >> 64);
    return (limb_t)p;
}

inline uint8_t addcarry(uint8_t c_in, limb_t a, limb_t b, limb_t* r) {
    unsigned __int128 s = (unsigned __int128)a + b + c_in;
    *r = (limb_t)s;
    return (uint8_t)(s >> 64);
}

inline uint8_t subborrow(uint8_t b_in, limb_t a, limb_t b, limb_t* r) {
    unsigned __int128 s = (unsigned __int128)a - b - b_in;
    *r = (limb_t)s;
    return (uint8_t)((limb_t)(s >> 64) & 1);
}

template<int N>
inline limb_t add(limb_t* r, const limb_t* a, const limb_t* b);

template<int N>
inline limb_t sub(limb_t* r, const limb_t* a, const limb_t* b);

template<int N>
inline limb_t addmul1(limb_t* r, const limb_t* a, limb_t b);

template<int N>
inline limb_t generic_add_block(limb_t* r, const limb_t* a, const limb_t* b, limb_t cy) {
    for (int i = 0; i < N; ++i)
        cy = addcarry((uint8_t)cy, a[i], b[i], r + i);
    return cy;
}

template<int N>
inline limb_t generic_sub_block(limb_t* r, const limb_t* a, const limb_t* b, limb_t bw) {
    for (int i = 0; i < N; ++i)
        bw = subborrow((uint8_t)bw, a[i], b[i], r + i);
    return bw;
}

template<int N>
inline limb_t generic_addmul1_block(limb_t* r, const limb_t* a, limb_t b, limb_t cy) {
    for (int i = 0; i < N; ++i) {
        unsigned __int128 acc = (unsigned __int128)a[i] * b + r[i] + cy;
        r[i] = (limb_t)acc;
        cy = (limb_t)(acc >> 64);
    }
    return cy;
}

template<int N>
inline limb_t generic_add(limb_t* r, const limb_t* a, const limb_t* b) {
    return generic_add_block<N>(r, a, b, 0);
}

template<int N>
inline limb_t generic_sub(limb_t* r, const limb_t* a, const limb_t* b) {
    return generic_sub_block<N>(r, a, b, 0);
}

template<int N>
inline limb_t generic_addmul1(limb_t* r, const limb_t* a, limb_t b) {
    return generic_addmul1_block<N>(r, a, b, 0);
}

#ifndef ZFACTOR_MPN_DECLARE_ONLY
template<int N>
inline limb_t add(limb_t* r, const limb_t* a, const limb_t* b) {
    return generic_add<N>(r, a, b);
}

template<int N>
inline limb_t sub(limb_t* r, const limb_t* a, const limb_t* b) {
    return generic_sub<N>(r, a, b);
}

template<int N>
inline limb_t addmul1(limb_t* r, const limb_t* a, limb_t b) {
    return generic_addmul1<N>(r, a, b);
}
#endif

} // namespace zfactor::mpn

#include "mpn_common.h"
