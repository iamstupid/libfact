#pragma once
#include <cstdint>
#include "intrin.h"
#include "popcnt.h"

namespace zfactor{
namespace fast_sieve{
    const uint8_t sieve_pos[]={1,7,11,13,17,19,23,29};
    const uint8_t sieve_idx[30] = {
        0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1,    0xFF, 0xFF,
        0xFF, 2,    0xFF, 3,    0xFF, 0xFF, 0xFF, 4,    0xFF, 5,
        0xFF, 0xFF, 0xFF, 6,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 7,
    };
    constexpr uint16_t sp_size = 64 * (11+13+17+19+23+29);

    struct alignas(64) small_pattern{
        uint8_t patterns[sp_size];
        const uint16_t small_primes[6]={11,13,17,19,23,29};
        uint16_t small_primes_idx[6];
        small_pattern(){
            small_primes_idx[0]=0;
            memset(patterns,0,sizeof patterns);
            for(int i=1;i<6;++i)
                small_primes_idx[i] = small_primes_idx[i-1] + small_primes[i-1];
            for(int i=0;i<6;++i){
                uint8_t* arr = patterns + small_primes_idx[i]*64;
                int k = small_primes[i];
                int max_k = k * 30 * 64;
                for(int j=k;j<max_k;j+=k){
                    int cind = j/30;
                    int cbind = j%30;
                    if(sieve_idx[cbind] != 0xFF)
                        arr[cind] |= 1u << sieve_idx[cbind];
                }
            }
        }
    };

    inline const small_pattern* get_small_pattern(){
        static const small_pattern pattern;
        return &pattern;
    }

    using small_pat_vector = 

    inline void sieve_small(uint8_t* per_prime_offset, )
}
}