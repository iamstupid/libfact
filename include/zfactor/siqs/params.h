#pragma once

// SIQS parameter selection and Knuth-Schroeppel multiplier.
//
// The parameter table is interpolated from YAFU/msieve empirical data.
// The multiplier selection follows Silverman's variant of Knuth-Schroeppel:
// score each small squarefree k by the expected average log contribution of
// the first few hundred primes to sieve values of k*n, minus a penalty of
// log(k)/2 for the increased number size.

#include <cmath>
#include <cstdint>
#include <algorithm>

#include "zfactor/fixint/uint.h"
#include "zfactor/jacobi.h"

namespace zfactor::siqs {

struct SiqsParams {
    uint32_t bits;          // input size this row applies to
    uint32_t fb_size;       // number of factor base primes
    uint32_t lp_mult;       // large_prime_bound = lp_mult * largest_fb_prime
    uint32_t num_blocks;    // sieve blocks per side (total sieve = 2 * num_blocks * 32768)
    double   dlp_exp;       // double-LP cutoff = lp_bound ^ dlp_exp
};

// Table from YAFU (non-AVX512 path), extended with small entries.
// Interpolated linearly by bit count of k*n.
inline constexpr SiqsParams param_table[] = {
    {  50,      30,   30,  1, 1.8 },
    {  60,      36,   40,  1, 1.8 },
    {  70,      50,   40,  1, 1.8 },
    {  80,      80,   40,  1, 1.8 },
    {  90,     120,   40,  1, 1.8 },
    { 100,     175,   50,  2, 1.8 },
    { 110,     275,   50,  2, 1.8 },
    { 120,     375,   50,  2, 1.8 },
    { 140,     828,   50,  2, 1.8 },
    { 149,    1028,   60,  2, 1.8 },
    { 165,    1228,   60,  2, 1.8 },
    { 181,    2247,   70,  2, 1.8 },
    { 198,    3485,   70,  4, 1.8 },
    { 215,    6357,   80,  4, 1.8 },
    { 232,   12132,   80,  6, 1.75},
    { 248,   26379,   90,  8, 1.75},
    { 265,   47158,   90, 10, 1.75},
    { 281,   60650,  100, 12, 1.8 },
    { 298,   71768,  120, 12, 1.8 },
    { 310,   86071,  120, 14, 1.85},
    { 320,   99745,  140, 16, 1.85},
    { 330,  115500,  150, 16, 1.85},
    { 340,  139120,  150, 18, 1.85},
    { 350,  166320,  150, 20, 1.85},
    { 360,  199584,  150, 22, 1.85},
    { 370,  239500,  150, 24, 1.9 },
    { 380,  287400,  175, 26, 1.9 },
    { 390,  344881,  175, 28, 1.9 },
    { 400,  413857,  175, 30, 1.9 },
};

inline constexpr int NUM_PARAM_ROWS = sizeof(param_table) / sizeof(param_table[0]);

inline SiqsParams get_params(uint32_t bits) {
    if (bits <= param_table[0].bits)
        return param_table[0];

    for (int i = 0; i < NUM_PARAM_ROWS - 1; i++) {
        if (bits > param_table[i].bits && bits <= param_table[i+1].bits) {
            double scale = double(param_table[i+1].bits - bits) /
                           double(param_table[i+1].bits - param_table[i].bits);
            SiqsParams p;
            p.bits = bits;
            p.fb_size = uint32_t(param_table[i+1].fb_size -
                        scale * (param_table[i+1].fb_size - param_table[i].fb_size) + 0.5);
            // Round fb_size up to multiple of 16 for SIMD alignment
            p.fb_size = (p.fb_size + 15) & ~15u;
            p.lp_mult = (param_table[i].lp_mult + param_table[i+1].lp_mult + 1) / 2;
            p.num_blocks = (param_table[i].num_blocks + param_table[i+1].num_blocks + 1) / 2;
            p.dlp_exp = (param_table[i].dlp_exp + param_table[i+1].dlp_exp) / 2.0;
            return p;
        }
    }

    // Extrapolate beyond table
    auto& last = param_table[NUM_PARAM_ROWS - 1];
    auto& prev = param_table[NUM_PARAM_ROWS - 2];
    double slope_fb = double(last.fb_size - prev.fb_size) / double(last.bits - prev.bits);
    double slope_bl = double(last.num_blocks - prev.num_blocks) / double(last.bits - prev.bits);
    SiqsParams p;
    p.bits = bits;
    p.fb_size = uint32_t(last.fb_size + slope_fb * (bits - last.bits));
    p.fb_size = (p.fb_size + 15) & ~15u;
    p.lp_mult = last.lp_mult;
    p.num_blocks = uint32_t(last.num_blocks + slope_bl * (bits - last.bits));
    p.dlp_exp = last.dlp_exp;
    return p;
}

// Compute kn mod p for small prime p, where kn = k * n (represented as UInt<N>).
template<int N>
inline uint32_t uint_mod_u32(const fixint::UInt<N>& a, uint32_t p) {
    // Horner's rule on 64-bit limbs: r = sum(d[i] * 2^(64i)) mod p
    uint64_t r = 0;
    for (int i = N - 1; i >= 0; i--) {
        // r = (r * 2^64 + d[i]) mod p
        // Use __int128 to avoid overflow
        unsigned __int128 w = (unsigned __int128)r << 64 | a.d[i];
        r = (uint64_t)(w % p);
    }
    return (uint32_t)r;
}

// Knuth-Schroeppel multiplier selection.
// Tests squarefree multipliers k and picks the one maximizing the expected
// contribution of small factor base primes to sieve values.
//
// Score(k) = -log(k)/2 + sum over small primes p of:
//   log(p)/(p-1) * { 2 if (kn/p) = 1, 1 if p | kn, 0 otherwise }
// Plus a bonus from 2 based on kn mod 8.
template<int N>
inline uint32_t select_multiplier(const fixint::UInt<N>& n) {
    static constexpr uint8_t mult_list[] = {
        1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19,
        21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38,
        39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59,
        61, 62, 65, 66, 67, 69, 70, 71, 73
    };
    static constexpr int NUM_MULT = sizeof(mult_list) / sizeof(mult_list[0]);
    static constexpr int NUM_TEST_PRIMES = 300;

    // Small primes for scoring
    static constexpr uint16_t small_primes[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
        157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
        239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
        331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
        421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
        509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
        613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
        709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
        821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
        919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
        1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
        1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181,
        1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277,
        1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361,
        1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451,
        1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531,
        1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609,
        1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699,
        1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789,
        1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889,
        1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997
    };
    int num_primes = std::min((int)(sizeof(small_primes)/sizeof(small_primes[0])),
                              NUM_TEST_PRIMES);

    // Precompute n mod p for each test prime
    uint32_t n_mod[NUM_TEST_PRIMES];
    for (int i = 0; i < num_primes; i++)
        n_mod[i] = uint_mod_u32<N>(n, small_primes[i]);

    double best_score = 1e30;
    uint8_t best_mult = 1;

    for (int m = 0; m < NUM_MULT; m++) {
        uint32_t k = mult_list[m];
        double score = 0.5 * std::log((double)k);

        // Contribution of 2
        uint32_t kn_mod8 = (uint32_t)((uint64_t)k * (n.d[0] & 7)) & 7;
        if (kn_mod8 == 1) score -= 2.0 * M_LN2;
        else if (kn_mod8 == 5) score -= M_LN2;
        else if (kn_mod8 == 3 || kn_mod8 == 7) score -= 0.5 * M_LN2;

        // Contribution of odd primes
        for (int i = 1; i < num_primes; i++) {
            uint32_t p = small_primes[i];
            double contrib = std::log((double)p) / (p - 1);
            uint32_t kn_modp = (uint32_t)(((uint64_t)n_mod[i] * k) % p);

            if (kn_modp == 0) {
                score -= contrib;  // p | kn: one root only
            } else if (jacobi_u64(kn_modp, p) == 1) {
                score -= 2.0 * contrib;  // two roots
            }
        }

        if (score < best_score) {
            best_score = score;
            best_mult = k;
        }
    }

    return best_mult;
}

} // namespace zfactor::siqs
