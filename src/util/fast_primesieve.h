#pragma once
#include <array>
#include <cstdint>
#include <stdint.h>
#include <vector>
#include <iterator>
#include <algorithm>
#include "intrin.h"
#include "popcnt.h"
#include "util/intrin.h"

namespace zfactor{
namespace fast_sieve{
    using namespace std;
    constexpr uint32_t mini_block_size = 16384;
    constexpr uint32_t macro_block_size = mini_block_size * 256;

    const uint8_t sieve_pos[]={1,7,11,13,17,19,23,29};
    const uint8_t sieve_idx[30] = {
        0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1,    0xFF, 0xFF,
        0xFF, 2,    0xFF, 3,    0xFF, 0xFF, 0xFF, 4,    0xFF, 5,
        0xFF, 0xFF, 0xFF, 6,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 7,
    };
    static constexpr uint8_t small_primes[] = {
        7,11,13,17,19,23,29,31,37,41,43,47,53,59,
        61,67,71,73,79,83,89,97,101,103,107,109,113,
        127,131,137,139,149,151,157,163,167,173,179,181,
        191,193,197,199,211,223,227,229
    };
    // We sieve small primes with a pattern sieve
    constexpr uint32_t n_small_primes = size(small_primes);
    constexpr uint32_t n_small_fold = 6;
    constexpr uint32_t n_patterns = n_small_primes - n_small_fold/2;
    constexpr uint32_t compute_pattern_size_total(){
        uint32_t total = 0;
        uint32_t safety_margin = vec_size - 1;
        // allocate vec_size - 1 safety margin after each prime pattern
        for(uint32_t i=0;i<n_small_fold-1-i;++i)
            total += small_primes[i] * small_primes[n_small_fold-1-i] + safety_margin;
        for(uint32_t i=n_small_fold;i<n_small_primes;++i)
            total += small_primes[i] + safety_margin;
        return total;
    }
    constexpr uint32_t pattern_size = compute_pattern_size_total();
    // small prime patterns
    struct alignas(64) small_table{
        uint8_t pattern[pattern_size];
        uint8_t length[n_patterns];
        uint16_t offset[n_patterns];
        small_table(){
            constexpr uint32_t safety_margin = vec_size - 1;
            memset(pattern, 0xFF, sizeof(pattern));
            offset[0] = 0;
            for(uint32_t i=0;i<n_small_primes;++i){
                uint32_t pat_size = (i<n_small_fold ? small_primes[i] * small_primes[n_small_fold - 1 - i] : small_primes[i]);
                uint32_t max_pat = (pat_size+safety_margin)*30;
                uint32_t pat_ind = i<n_small_fold ? min(i, n_small_fold - 1 - i) : i - n_small_fold/2;
                length[pat_ind] = pat_size;
                if(i && (i<n_small_fold/2 || i>=n_small_fold))
                    offset[pat_ind] = offset[pat_ind-1] + length[pat_ind-1] + safety_margin;
                uint8_t* pat = pattern + offset[pat_ind];
                for(uint32_t j = small_primes[i]; j < max_pat; j += small_primes[i]){
                    uint32_t limb_ind = j / 30;
                    uint8_t limb_bit = sieve_idx[j % 30];
                    if(limb_bit != 0xFF)
                        pat[limb_ind] &= ~(1u << limb_bit);
                }
            }
        }
    };
    inline const small_table* get_small_table(){
        static const small_table st;
        return &st;
    }
    // sieve small:
    inline void sieve_small();
}
}