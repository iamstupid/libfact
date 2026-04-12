#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace zfactor::fixint::mpn {

extern "C" void zfactor_fixint_sqr_diag_addlsh1_x86_64(limb_t* rp,
                                                       const limb_t* tp,
                                                       const limb_t* up,
                                                       std::size_t n);

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
inline uint8_t lshift1(limb_t* r, const limb_t* a) {
    uint8_t carry = 0;
    for (int i = 0; i < N; ++i) {
        limb_t v = a[i];
        r[i] = (v << 1) | carry;
        carry = static_cast<uint8_t>(v >> 63);
    }
    return carry;
}

template<int N>
inline uint8_t rshift1(limb_t* r, const limb_t* a) {
    uint8_t carry = 0;
    for (int i = N - 1; i >= 0; --i) {
        limb_t v = a[i];
        r[i] = (v >> 1) | (limb_t(carry) << 63);
        carry = static_cast<uint8_t>(v & 1);
    }
    return carry;
}

template<int N>
inline bool lshift_small_simd(limb_t* r, const limb_t* a, unsigned bits) {
#if defined(__AVX2__)
    if constexpr (N >= 4) {
        if (r != a) {
            const __m128i cnt = _mm_cvtsi64_si128(bits);
            const __m128i inv = _mm_cvtsi64_si128(64u - bits);
            int i = 1;
            for (; i + 4 <= N; i += 4) {
                auto cur = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                auto prev = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i - 1));
                auto lo = _mm256_sll_epi64(cur, cnt);
                auto hi = _mm256_srl_epi64(prev, inv);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), _mm256_or_si256(lo, hi));
            }
#if defined(__SSE2__) || defined(_M_X64)
            for (; i + 2 <= N; i += 2) {
                auto cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
                auto prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i - 1));
                auto lo = _mm_sll_epi64(cur, cnt);
                auto hi = _mm_srl_epi64(prev, inv);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(r + i), _mm_or_si128(lo, hi));
            }
#endif
            for (; i < N; ++i)
                r[i] = (a[i] << bits) | (a[i - 1] >> (64 - bits));
            r[0] = a[0] << bits;
            return true;
        }
    }
#endif
#if defined(__SSE2__) || defined(_M_X64)
    if constexpr (N >= 2) {
        if (r != a) {
            const __m128i cnt = _mm_cvtsi64_si128(bits);
            const __m128i inv = _mm_cvtsi64_si128(64u - bits);
            int i = 1;
            for (; i + 2 <= N; i += 2) {
                auto cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
                auto prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i - 1));
                auto lo = _mm_sll_epi64(cur, cnt);
                auto hi = _mm_srl_epi64(prev, inv);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(r + i), _mm_or_si128(lo, hi));
            }
            for (; i < N; ++i)
                r[i] = (a[i] << bits) | (a[i - 1] >> (64 - bits));
            r[0] = a[0] << bits;
            return true;
        }
    }
#endif
    return false;
}

template<int N>
inline bool rshift_small_simd(limb_t* r, const limb_t* a, unsigned bits) {
#if defined(__AVX2__)
    if constexpr (N >= 5) {
        if (r != a) {
            const __m128i cnt = _mm_cvtsi64_si128(bits);
            const __m128i inv = _mm_cvtsi64_si128(64u - bits);
            int i = 0;
            for (; i + 4 < N; i += 4) {
                auto cur = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                auto next = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 1));
                auto lo = _mm256_srl_epi64(cur, cnt);
                auto hi = _mm256_sll_epi64(next, inv);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + i), _mm256_or_si256(lo, hi));
            }
#if defined(__SSE2__) || defined(_M_X64)
            for (; i + 2 < N; i += 2) {
                auto cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
                auto next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 1));
                auto lo = _mm_srl_epi64(cur, cnt);
                auto hi = _mm_sll_epi64(next, inv);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(r + i), _mm_or_si128(lo, hi));
            }
#endif
            for (; i + 1 < N; ++i)
                r[i] = (a[i] >> bits) | (a[i + 1] << (64 - bits));
            r[N - 1] = a[N - 1] >> bits;
            return true;
        }
    }
#endif
#if defined(__SSE2__) || defined(_M_X64)
    if constexpr (N >= 3) {
        if (r != a) {
            const __m128i cnt = _mm_cvtsi64_si128(bits);
            const __m128i inv = _mm_cvtsi64_si128(64u - bits);
            int i = 0;
            for (; i + 2 < N; i += 2) {
                auto cur = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
                auto next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 1));
                auto lo = _mm_srl_epi64(cur, cnt);
                auto hi = _mm_sll_epi64(next, inv);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(r + i), _mm_or_si128(lo, hi));
            }
            for (; i + 1 < N; ++i)
                r[i] = (a[i] >> bits) | (a[i + 1] << (64 - bits));
            r[N - 1] = a[N - 1] >> bits;
            return true;
        }
    }
#endif
    return false;
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
    if (limb_shift == 0) {
        if (lshift_small_simd<N>(r, a, bit_shift))
            return;
        lshift_small<N>(r, a, bit_shift);
        return;
    }
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
    if (limb_shift == 0) {
        if (rshift_small_simd<N>(r, a, bit_shift))
            return;
        rshift_small<N>(r, a, bit_shift);
        return;
    }
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
inline void bitand_impl(limb_t* r, const limb_t* a, const limb_t* b) {
#if defined(__AVX2__)
    if constexpr (N >= 4) {
        constexpr int full = N / 4;
        constexpr int rem = N % 4;
        for (int j = 0; j < full; ++j) {
            auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j * 4));
            auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j * 4));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + j * 4), _mm256_and_si256(va, vb));
        }
        if constexpr (rem != 0)
            bitand_impl<rem>(r + full * 4, a + full * 4, b + full * 4);
    } else
#endif
#if defined(__SSE2__) || defined(_M_X64)
    if constexpr (N >= 2) {
        constexpr int full = N / 2;
        constexpr int rem = N % 2;
        for (int j = 0; j < full; ++j) {
            auto va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j * 2));
            auto vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j * 2));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(r + j * 2), _mm_and_si128(va, vb));
        }
        if constexpr (rem != 0)
            r[full * 2] = a[full * 2] & b[full * 2];
    } else
#endif
    {
        for (int i = 0; i < N; ++i)
            r[i] = a[i] & b[i];
    }
}

template<int N>
inline void bitor_impl(limb_t* r, const limb_t* a, const limb_t* b) {
#if defined(__AVX2__)
    if constexpr (N >= 4) {
        constexpr int full = N / 4;
        constexpr int rem = N % 4;
        for (int j = 0; j < full; ++j) {
            auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j * 4));
            auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j * 4));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + j * 4), _mm256_or_si256(va, vb));
        }
        if constexpr (rem != 0)
            bitor_impl<rem>(r + full * 4, a + full * 4, b + full * 4);
    } else
#endif
#if defined(__SSE2__) || defined(_M_X64)
    if constexpr (N >= 2) {
        constexpr int full = N / 2;
        constexpr int rem = N % 2;
        for (int j = 0; j < full; ++j) {
            auto va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j * 2));
            auto vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j * 2));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(r + j * 2), _mm_or_si128(va, vb));
        }
        if constexpr (rem != 0)
            r[full * 2] = a[full * 2] | b[full * 2];
    } else
#endif
    {
        for (int i = 0; i < N; ++i)
            r[i] = a[i] | b[i];
    }
}

template<int N>
inline void bitxor_impl(limb_t* r, const limb_t* a, const limb_t* b) {
#if defined(__AVX2__)
    if constexpr (N >= 4) {
        constexpr int full = N / 4;
        constexpr int rem = N % 4;
        for (int j = 0; j < full; ++j) {
            auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j * 4));
            auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j * 4));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + j * 4), _mm256_xor_si256(va, vb));
        }
        if constexpr (rem != 0)
            bitxor_impl<rem>(r + full * 4, a + full * 4, b + full * 4);
    } else
#endif
#if defined(__SSE2__) || defined(_M_X64)
    if constexpr (N >= 2) {
        constexpr int full = N / 2;
        constexpr int rem = N % 2;
        for (int j = 0; j < full; ++j) {
            auto va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j * 2));
            auto vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + j * 2));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(r + j * 2), _mm_xor_si128(va, vb));
        }
        if constexpr (rem != 0)
            r[full * 2] = a[full * 2] ^ b[full * 2];
    } else
#endif
    {
        for (int i = 0; i < N; ++i)
            r[i] = a[i] ^ b[i];
    }
}

template<int N>
inline void bitnot_impl(limb_t* r, const limb_t* a) {
#if defined(__AVX2__)
    if constexpr (N >= 4) {
        constexpr int full = N / 4;
        constexpr int rem = N % 4;
        const auto ones = _mm256_set1_epi64x(-1);
        for (int j = 0; j < full; ++j) {
            auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j * 4));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(r + j * 4), _mm256_xor_si256(va, ones));
        }
        if constexpr (rem != 0)
            bitnot_impl<rem>(r + full * 4, a + full * 4);
    } else
#endif
#if defined(__SSE2__) || defined(_M_X64)
    if constexpr (N >= 2) {
        constexpr int full = N / 2;
        constexpr int rem = N % 2;
        const auto ones = _mm_set1_epi64x(-1);
        for (int j = 0; j < full; ++j) {
            auto va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + j * 2));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(r + j * 2), _mm_xor_si128(va, ones));
        }
        if constexpr (rem != 0)
            r[full * 2] = ~a[full * 2];
    } else
#endif
    {
        for (int i = 0; i < N; ++i)
            r[i] = ~a[i];
    }
}

// Branchless conditional subtract: r = (a >= b) ? a - b : a
template<int N>
inline void csub(limb_t* r, const limb_t* a, const limb_t* b) {
    limb_t tmp[N];
    limb_t bw = sub<N>(tmp, a, b);
    limb_t mask = -bw;
    for (int i = 0; i < N; ++i)
        r[i] = (a[i] & mask) | (tmp[i] & ~mask);
}

// Branchless conditional add: r = a + (flag ? b : 0)
template<int N>
inline void cadd(limb_t* r, const limb_t* a, const limb_t* b, limb_t flag) {
    limb_t mask = -flag;
    limb_t masked[N];
    for (int i = 0; i < N; ++i)
        masked[i] = b[i] & mask;
    add<N>(r, a, masked);
}

template<int N, int I>
inline void sqr_cross_terms(limb_t* tp, const limb_t* a) {
    if constexpr (I == 0) {
        tp[N - 1] = addmul1<N - 1>(tp, a + 1, a[0]);
        sqr_cross_terms<N, 1>(tp, a);
    } else if constexpr (I + 1 < N) {
        tp[N + I - 1] = addmul1<N - I - 1>(tp + 2 * I, a + I + 1, a[I]);
        sqr_cross_terms<N, I + 1>(tp, a);
    }
}

template<int N, int I>
inline void sqr_diag_rows(limb_t* r, const limb_t* a) {
    if constexpr (I < N) {
        limb_t hi = 0;
        r[2 * I] = umul128(a[I], a[I], &hi);
        r[2 * I + 1] = hi;
        sqr_diag_rows<N, I + 1>(r, a);
    }
}

template<int N>
inline uint8_t addlsh1(limb_t* r, const limb_t* a) {
    uint8_t shift = 0;
    uint8_t cy = 0;
    for (int i = 0; i < N; ++i) {
        limb_t v = (a[i] << 1) | limb_t(shift);
        shift = static_cast<uint8_t>(a[i] >> 63);
        cy = addcarry(cy, r[i], v, r + i);
    }
    return static_cast<uint8_t>(cy + shift);
}

template<int N>
inline bool sqr_diag_addlsh1_asm(limb_t* r, const limb_t* tp, const limb_t* a) {
#if defined(__x86_64__) || defined(_M_X64)
    zfactor_fixint_sqr_diag_addlsh1_x86_64(r, tp, a, std::size_t(N));
    return true;
#else
    (void)r;
    (void)tp;
    (void)a;
    return false;
#endif
}

template<int N>
inline void sqr_impl(limb_t* r, const limb_t* a) {
    if constexpr (N == 1) {
        limb_t hi = 0;
        r[0] = umul128(a[0], a[0], &hi);
        r[1] = hi;
    } else {
        limb_t tp[2 * N - 2];
        set_zero<2 * N - 2>(tp);
        sqr_cross_terms<N, 0>(tp, a);
        if (sqr_diag_addlsh1_asm<N>(r, tp, a))
            return;
        sqr_diag_rows<N, 0>(r, a);
        r[2 * N - 1] += addlsh1<2 * N - 2>(r + 1, tp);
    }
}

template<int N>
inline void bitand_(limb_t* r, const limb_t* a, const limb_t* b) {
    bitand_impl<N>(r, a, b);
}

template<int N>
inline void bitor_(limb_t* r, const limb_t* a, const limb_t* b) {
    bitor_impl<N>(r, a, b);
}

template<int N>
inline void bitxor_(limb_t* r, const limb_t* a, const limb_t* b) {
    bitxor_impl<N>(r, a, b);
}

template<int N>
inline void bitnot_(limb_t* r, const limb_t* a) {
    bitnot_impl<N>(r, a);
}

template<int N>
inline uint8_t addmul(limb_t* r, const limb_t* a, const limb_t* b) {
    uint8_t overflow = 0;
    for (int j = 0; j < N; ++j) {
        limb_t cy = addmul1<N>(r + j, a, b[j]);
        int k = j + N;
        uint8_t carry = addcarry(0, r[k], cy, r + k);
        ++k;
        while (carry != 0 && k < 2 * N) {
            carry = addcarry(carry, r[k], 0, r + k);
            ++k;
        }
        overflow |= carry;
    }
    return overflow;
}

template<int N, int J>
inline void mul_rows(limb_t* r, const limb_t* a, const limb_t* b) {
    if constexpr (J < N) {
        r[J + N] = addmul1<N>(r + J, a, b[J]);
        mul_rows<N, J + 1>(r, a, b);
    }
}

// Non-square multiply: r[N+M] = a[N] * b[M], using addmul1<N>.
template<int N, int M, int J>
inline void mul_nm_rows(limb_t* r, const limb_t* a, const limb_t* b) {
    if constexpr (J < M) {
        r[J + N] = addmul1<N>(r + J, a, b[J]);
        mul_nm_rows<N, M, J + 1>(r, a, b);
    }
}

template<int N, int M>
inline void mul_nm(limb_t* r, const limb_t* a, const limb_t* b) {
    set_zero<N + M>(r);
    mul_nm_rows<N, M, 0>(r, a, b);
}

// Low-half multiply: r[0..N-1] = (a * b) mod 2^(64N)
// ~N²/2 multiplications — only computes limbs that contribute to positions 0..N-1.
template<int N, int J>
inline void mullow_rows(limb_t* r, const limb_t* a, const limb_t* b) {
    if constexpr (J < N) {
        addmul1<N - J>(r + J, a, b[J]);
        mullow_rows<N, J + 1>(r, a, b);
    }
}

template<int N>
inline void mullow(limb_t* r, const limb_t* a, const limb_t* b) {
    set_zero<N>(r);
    mullow_rows<N, 0>(r, a, b);
}

template<int N>
inline void mul(limb_t* r, const limb_t* a, const limb_t* b) {
    set_zero<2 * N>(r);
    mul_rows<N, 0>(r, a, b);
}

template<int N>
inline void sqr(limb_t* r, const limb_t* a) {
    if constexpr (N == 1) {
        mul<N>(r, a, a);
        return;
    }
    if constexpr (N <= 4) {
        if (sqr_small_asm<N>(r, a))
            return;
    }
    sqr_impl<N>(r, a);
}

// ============================================================================
// Runtime-n dispatchers for compile-time-N specializations.
// Useful for code that operates on bounded but runtime-sized slices
// (e.g., Lehmer GCD apply_matrix, basecase divrem).
// ============================================================================

inline limb_t addmul1_n(limb_t* r, const limb_t* a, limb_t b, int n) {
    switch (n) {
        case 1: return addmul1<1>(r, a, b);
        case 2: return addmul1<2>(r, a, b);
        case 3: return addmul1<3>(r, a, b);
        case 4: return addmul1<4>(r, a, b);
        case 5: return addmul1<5>(r, a, b);
        case 6: return addmul1<6>(r, a, b);
        case 7: return addmul1<7>(r, a, b);
        case 8: return addmul1<8>(r, a, b);
    }
    limb_t cy = 0;
    for (int i = 0; i < n; ++i) {
        unsigned __int128 p = (unsigned __int128)a[i] * b + r[i] + cy;
        r[i] = static_cast<limb_t>(p);
        cy = static_cast<limb_t>(p >> 64);
    }
    return cy;
}

inline limb_t submul1_n(limb_t* r, const limb_t* a, limb_t b, int n) {
    switch (n) {
        case 1: return submul1<1>(r, a, b);
        case 2: return submul1<2>(r, a, b);
        case 3: return submul1<3>(r, a, b);
        case 4: return submul1<4>(r, a, b);
        case 5: return submul1<5>(r, a, b);
        case 6: return submul1<6>(r, a, b);
        case 7: return submul1<7>(r, a, b);
        case 8: return submul1<8>(r, a, b);
    }
    limb_t bw = 0;
    for (int i = 0; i < n; ++i) {
        unsigned __int128 p = (unsigned __int128)a[i] * b + bw;
        limb_t lo = static_cast<limb_t>(p);
        bw = static_cast<limb_t>(p >> 64) + (r[i] < lo);
        r[i] -= lo;
    }
    return bw;
}

inline uint8_t sub_n(limb_t* r, const limb_t* a, const limb_t* b, int n) {
    switch (n) {
        case 1: return sub<1>(r, a, b);
        case 2: return sub<2>(r, a, b);
        case 3: return sub<3>(r, a, b);
        case 4: return sub<4>(r, a, b);
        case 5: return sub<5>(r, a, b);
        case 6: return sub<6>(r, a, b);
        case 7: return sub<7>(r, a, b);
        case 8: return sub<8>(r, a, b);
    }
    uint8_t bw = 0;
    for (int i = 0; i < n; ++i)
        bw = subborrow(bw, a[i], b[i], &r[i]);
    return bw;
}

// ============================================================================
// Schoolbook long division (Knuth Algorithm D)
// u[0..N-1] / v[0..N-1] -> q[0..N-1], r[0..N-1]
// Handles variable effective sizes internally.
// ============================================================================

namespace detail_div {

// Core schoolbook divrem. Operates on u[0..un-1] / v[0..vn-1].
// Writes quotient to q (caller zeros it, at least un-vn+1 limbs available)
// and remainder to r (caller zeros it, at least vn limbs available).
// MAXBUF: compile-time max limbs for stack buffers.
template<int MAXBUF>
inline void schoolbook_div(limb_t* q, limb_t* r,
                           const limb_t* u_in, int un,
                           const limb_t* v_in, int vn) {
    // Trim leading zeros
    while (un > 0 && u_in[un - 1] == 0) --un;
    while (vn > 0 && v_in[vn - 1] == 0) --vn;

    if (vn == 0 || un == 0) return;

    // u < v → q=0, r=u
    if (un < vn) {
        for (int i = 0; i < un; ++i) r[i] = u_in[i];
        return;
    }
    if (un == vn) {
        for (int i = un - 1; i >= 0; --i) {
            if (u_in[i] < v_in[i]) {
                for (int j = 0; j < un; ++j) r[j] = u_in[j];
                return;
            }
            if (u_in[i] > v_in[i]) break;
        }
    }

    // Single-limb divisor: fast path
    if (vn == 1) {
        limb_t rem = 0;
        for (int i = un - 1; i >= 0; --i) {
            unsigned __int128 cur = ((unsigned __int128)rem << 64) | u_in[i];
            q[i] = static_cast<limb_t>(cur / v_in[0]);
            rem = static_cast<limb_t>(cur % v_in[0]);
        }
        r[0] = rem;
        return;
    }

    // --- Knuth Algorithm D (vn >= 2) ---

    // D1: Normalize
    unsigned shift = static_cast<unsigned>(std::countl_zero(v_in[vn - 1]));

    limb_t uu[MAXBUF + 1] = {};
    limb_t vv[MAXBUF] = {};

    if (shift > 0) {
        for (int i = vn - 1; i > 0; --i)
            vv[i] = (v_in[i] << shift) | (v_in[i - 1] >> (64 - shift));
        vv[0] = v_in[0] << shift;

        uu[un] = u_in[un - 1] >> (64 - shift);
        for (int i = un - 1; i > 0; --i)
            uu[i] = (u_in[i] << shift) | (u_in[i - 1] >> (64 - shift));
        uu[0] = u_in[0] << shift;
    } else {
        for (int i = 0; i < vn; ++i) vv[i] = v_in[i];
        for (int i = 0; i < un; ++i) uu[i] = u_in[i];
    }

    // D2-D6: Main loop
    for (int j = un - vn; j >= 0; --j) {
        // D3: Estimate q_hat
        limb_t q_hat;
        if (uu[j + vn] >= vv[vn - 1]) {
            q_hat = UINT64_MAX;
        } else {
            unsigned __int128 num = ((unsigned __int128)uu[j + vn] << 64) | uu[j + vn - 1];
            q_hat = static_cast<limb_t>(num / vv[vn - 1]);
            limb_t r_hat = static_cast<limb_t>(num % vv[vn - 1]);

            // Refine: at most 2 iterations
            for (;;) {
                unsigned __int128 lhs = (unsigned __int128)q_hat * vv[vn - 2];
                unsigned __int128 rhs = ((unsigned __int128)r_hat << 64) | uu[j + vn - 2];
                if (lhs <= rhs) break;
                --q_hat;
                r_hat += vv[vn - 1];
                if (r_hat < vv[vn - 1]) break; // overflow → done
            }
        }

        // D4: Multiply and subtract
        // Compute prod = q_hat * vv (vn+1 limbs), then subtract from uu[j..j+vn]
        limb_t prod[MAXBUF + 1] = {};
        {
            limb_t cy = 0;
            for (int i = 0; i < vn; ++i) {
                unsigned __int128 p = (unsigned __int128)q_hat * vv[i] + cy;
                prod[i] = static_cast<limb_t>(p);
                cy = static_cast<limb_t>(p >> 64);
            }
            prod[vn] = cy;
        }

        uint8_t bw = 0;
        for (int i = 0; i <= vn; ++i)
            bw = subborrow(bw, uu[j + i], prod[i], &uu[j + i]);

        // D5: If oversubtracted, add back
        if (bw) {
            --q_hat;
            uint8_t cy = 0;
            for (int i = 0; i < vn; ++i)
                cy = addcarry(cy, uu[j + i], vv[i], &uu[j + i]);
            uu[j + vn] += cy;
        }

        // D6
        q[j] = q_hat;
    }

    // D7: Unnormalize remainder
    if (shift > 0) {
        for (int i = 0; i < vn - 1; ++i)
            r[i] = (uu[i] >> shift) | (uu[i + 1] << (64 - shift));
        r[vn - 1] = uu[vn - 1] >> shift;
    } else {
        for (int i = 0; i < vn; ++i) r[i] = uu[i];
    }
}

} // namespace detail_div

#if defined(__x86_64__)
extern "C" void zfactor_mg_divrem(
    limb_t* qp, limb_t* rp,
    const limb_t* up, std::int64_t un,
    const limb_t* vp, std::int64_t vn);
#endif

// Thin wrapper around the LLVM-IR-generated all-in-one MG divrem kernel.
// Handles trim and short-circuit cases; passes raw inputs to the kernel.
template<int MAXBUF>
inline void mg_divrem(limb_t* q, limb_t* r,
                      const limb_t* u_in, int un,
                      const limb_t* v_in, int vn) {
    while (un > 0 && u_in[un - 1] == 0) --un;
    while (vn > 0 && v_in[vn - 1] == 0) --vn;
    if (vn == 0 || un == 0) return;

    if (un < vn) {
        for (int i = 0; i < un; ++i) r[i] = u_in[i];
        return;
    }
    if (un == vn) {
        for (int i = un - 1; i >= 0; --i) {
            if (u_in[i] < v_in[i]) {
                for (int j = 0; j < un; ++j) r[j] = u_in[j];
                return;
            }
            if (u_in[i] > v_in[i]) break;
        }
    }

#if defined(__x86_64__)
    zfactor_mg_divrem(q, r, u_in, un, v_in, vn);
#else
    detail_div::schoolbook_div<MAXBUF>(q, r, u_in, un, v_in, vn);
#endif
}

template<int N>
inline void divrem(limb_t* q, limb_t* r, const limb_t* a, const limb_t* b) {
    set_zero<N>(q);
    set_zero<N>(r);
    mg_divrem<N>(q, r, a, N, b, N);
}

template<int M, int N>
inline void divrem_wide(limb_t* q, limb_t* r, const limb_t* u, const limb_t* v) {
    static_assert(M >= N);
    for (int i = 0; i < M - N + 1; ++i) q[i] = 0;
    set_zero<N>(r);
    mg_divrem<M>(q, r, u, M, v, N);
}

} // namespace zfactor::fixint::mpn
