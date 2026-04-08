#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace zfactor::mpn {

constexpr unsigned limb_bits = 64;

template<int N>
inline void set_zero(limb_t* r) {
    static_assert(N >= 0);
    std::memset(r, 0, sizeof(limb_t) * N);
}

template<int N>
inline void copy(limb_t* r, const limb_t* a) {
    static_assert(N >= 0);
    std::memmove(r, a, sizeof(limb_t) * N);
}

template<int N>
inline bool is_zero(const limb_t* a) {
    limb_t acc = 0;
    for (int i = 0; i < N; ++i)
        acc |= a[i];
    return acc == 0;
}

template<int N>
inline int cmp(const limb_t* a, const limb_t* b) {
    for (int i = N - 1; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

template<int N>
inline limb_t add1(limb_t* r, const limb_t* a, limb_t b) {
    limb_t cy = addcarry(0, a[0], b, r);
    for (int i = 1; i < N; ++i)
        cy = addcarry((uint8_t)cy, a[i], 0, r + i);
    return cy;
}

template<int N>
inline limb_t sub1(limb_t* r, const limb_t* a, limb_t b) {
    limb_t bw = subborrow(0, a[0], b, r);
    for (int i = 1; i < N; ++i)
        bw = subborrow((uint8_t)bw, a[i], 0, r + i);
    return bw;
}

template<int N>
inline limb_t lshift1(limb_t* r, const limb_t* a) {
    limb_t carry = 0;
    for (int i = 0; i < N; ++i) {
        limb_t v = a[i];
        r[i] = (v << 1) | carry;
        carry = v >> 63;
    }
    return carry;
}

template<int N>
inline limb_t rshift1(limb_t* r, const limb_t* a) {
    limb_t carry = 0;
    for (int i = N - 1; i >= 0; --i) {
        limb_t v = a[i];
        r[i] = (v >> 1) | (carry << 63);
        carry = v & 1;
    }
    return carry;
}

template<int N>
inline void lshift(limb_t* r, const limb_t* a, unsigned bits) {
    if (bits == 0) {
        copy<N>(r, a);
        return;
    }
    if (bits >= unsigned(N * limb_bits)) {
        set_zero<N>(r);
        return;
    }
    unsigned limb_shift = bits / limb_bits;
    unsigned bit_shift = bits % limb_bits;
    for (int i = N - 1; i >= 0; --i) {
        limb_t out = 0;
        if (i >= (int)limb_shift) {
            out = a[i - limb_shift] << bit_shift;
            if (bit_shift != 0 && i > (int)limb_shift)
                out |= a[i - limb_shift - 1] >> (limb_bits - bit_shift);
        }
        r[i] = out;
    }
}

template<int N>
inline void rshift(limb_t* r, const limb_t* a, unsigned bits) {
    if (bits == 0) {
        copy<N>(r, a);
        return;
    }
    if (bits >= unsigned(N * limb_bits)) {
        set_zero<N>(r);
        return;
    }
    unsigned limb_shift = bits / limb_bits;
    unsigned bit_shift = bits % limb_bits;
    for (int i = 0; i < N; ++i) {
        limb_t out = 0;
        if (i + (int)limb_shift < N) {
            out = a[i + limb_shift] >> bit_shift;
            if (bit_shift != 0 && i + (int)limb_shift + 1 < N)
                out |= a[i + limb_shift + 1] << (limb_bits - bit_shift);
        }
        r[i] = out;
    }
}

template<int N>
inline unsigned clz(const limb_t* a) {
    for (int i = N - 1; i >= 0; --i) {
        if (a[i] != 0)
            return unsigned((N - 1 - i) * limb_bits + std::countl_zero(a[i]));
    }
    return unsigned(N * limb_bits);
}

template<int N>
inline unsigned ctz(const limb_t* a) {
    for (int i = 0; i < N; ++i) {
        if (a[i] != 0)
            return unsigned(i * limb_bits + std::countr_zero(a[i]));
    }
    return unsigned(N * limb_bits);
}

template<int N>
inline unsigned bit_length(const limb_t* a) {
    return unsigned(N * limb_bits) - clz<N>(a);
}

template<int N>
inline bool test_bit(const limb_t* a, std::size_t bit) {
    if (bit >= std::size_t(N) * limb_bits)
        return false;
    return (a[bit / limb_bits] >> (bit % limb_bits)) & 1u;
}

template<int N>
inline void set_bit(limb_t* a, std::size_t bit) {
    if (bit >= std::size_t(N) * limb_bits)
        return;
    a[bit / limb_bits] |= limb_t(1) << (bit % limb_bits);
}

template<int N>
inline void clear_bit(limb_t* a, std::size_t bit) {
    if (bit >= std::size_t(N) * limb_bits)
        return;
    a[bit / limb_bits] &= ~(limb_t(1) << (bit % limb_bits));
}

template<int N>
inline void bitand_(limb_t* r, const limb_t* a, const limb_t* b) {
    for (int i = 0; i < N; ++i)
        r[i] = a[i] & b[i];
}

template<int N>
inline void bitor_(limb_t* r, const limb_t* a, const limb_t* b) {
    for (int i = 0; i < N; ++i)
        r[i] = a[i] | b[i];
}

template<int N>
inline void bitxor_(limb_t* r, const limb_t* a, const limb_t* b) {
    for (int i = 0; i < N; ++i)
        r[i] = a[i] ^ b[i];
}

template<int N>
inline void bitnot_(limb_t* r, const limb_t* a) {
    for (int i = 0; i < N; ++i)
        r[i] = ~a[i];
}

template<int N>
inline limb_t addmul(limb_t* r, const limb_t* a, const limb_t* b) {
    limb_t overflow = 0;
    for (int j = 0; j < N; ++j) {
        limb_t cy = addmul1<N>(r + j, a, b[j]);
        int k = j + N;
        limb_t carry = addcarry(0, r[k], cy, r + k);
        ++k;
        while (carry != 0 && k < 2 * N) {
            carry = addcarry((uint8_t)carry, r[k], 0, r + k);
            ++k;
        }
        overflow |= carry;
    }
    return overflow;
}

template<int N>
inline void mul(limb_t* r, const limb_t* a, const limb_t* b) {
    set_zero<2 * N>(r);
    (void)addmul<N>(r, a, b);
}

template<int N>
inline void sqr(limb_t* r, const limb_t* a) {
    set_zero<2 * N>(r);
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            limb_t hi = 0;
            limb_t lo = umul128(a[i], a[j], &hi);
            int pos = i + j;
            uint8_t carry = addcarry(0, r[pos], lo, r + pos);
            ++pos;
            carry = addcarry(carry, r[pos], hi, r + pos);
            ++pos;
            while (carry != 0 && pos < 2 * N) {
                carry = addcarry(carry, r[pos], 0, r + pos);
                ++pos;
            }
        }
    }

    (void)lshift1<2 * N>(r, r);

    for (int i = 0; i < N; ++i) {
        limb_t hi = 0;
        limb_t lo = umul128(a[i], a[i], &hi);
        uint8_t carry = addcarry(0, r[2 * i], lo, r + 2 * i);
        carry = addcarry(carry, r[2 * i + 1], hi, r + 2 * i + 1);
        int pos = 2 * i + 2;
        while (carry != 0 && pos < 2 * N) {
            carry = addcarry(carry, r[pos], 0, r + pos);
            ++pos;
        }
    }
}

} // namespace zfactor::mpn
