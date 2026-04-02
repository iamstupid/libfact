#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <stdint.h>
#include <vector>
#include <iterator>
#include <algorithm>
#include "intrin.h"
#include "popcnt.h"
#include "block_vec.h"
#include "../third_party/libdivide.h"

namespace zfactor{
namespace fast_sieve{
    using namespace std;
    constexpr uint32_t mini_block_size = 16384;
    constexpr uint32_t mini_per_alloc = 64;
    constexpr uint32_t macro_block_size = mini_block_size * mini_per_alloc;

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
        191,193,197,199,211,223,227
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
        uint8_t length[n_patterns], ilen[n_patterns];
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
                ilen[pat_ind] = vec_size % pat_size;
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
    inline void sieve_small(uint8_t* sieve, uint8_t* small_offsets){
        memset(sieve, 0xFF, mini_block_size);
        const small_table* st = get_small_table();
        for(uint32_t i = 0; i < n_patterns; ++i){
            const uint8_t* pat = st->pattern + st->offset[i];
            uint16_t off = small_offsets[i];
            uint8_t len = st->length[i];
            uint8_t delta = st->ilen[i];
            for(uint16_t j = 0; j < mini_block_size; j += vec_size){
                (vec::load(sieve + j) & vec::load(pat + off)).store(sieve + j);
                off += delta;
                if(off >= len) off -= len;
            }
            small_offsets[i] = off;
        }
    }
    // sieve kernel
    struct sieve_kernel;

    struct sieve_ctx{
        static constexpr uint32_t n_bucklen = 1023;
        block_vec<uint8_t[mini_block_size], mini_per_alloc> sieve;
        uint64_t lower_bound, upper_bound, max_sieve, current_pos;
        // max_sieve is the sieving upbound
        // lower_bound and upper_bound are the bounds of current sieving window
        // each * 30 is the actual bound. i.e. [lower_bound*30, upper_bound*30)
        uint8_t small_offsets[n_patterns];
        block_vec<uint32_t, 1u<<18> prime_offsets;
        // prime starting offset relative to lower_bound
        struct bucket{ // make a bucket 4KiB
            uint32_t length, bucks[n_bucklen];
            void clear(){ length = 0; }
            void push_back(uint32_t t){ bucks[length++] = t; }
            bool is_full(){ return length >= n_bucklen; }
        };
        block_vec<bucket, mini_per_alloc << 2> buckets;
        void sieve_next();
        inline uint64_t pos_to_prime(uint64_t q){
            return (lower_bound + q/8)*30 + sieve_pos[q & 7];
        }
        uint64_t get_next_pos(){
            uint64_t total_bits = (uint64_t)sieve.size() * mini_block_size * 8;
            uint64_t pos = current_pos;
            if(pos >= total_bits) return total_bits;

            uint32_t byte_off = (uint32_t)(pos / 8);
            uint32_t block_idx = byte_off / mini_block_size;
            uint32_t in_block = byte_off % mini_block_size;
            uint32_t bit_off = (uint32_t)(pos & 7);

            for(; block_idx < sieve.size(); ++block_idx){
                const uint64_t* words = (const uint64_t*)sieve[block_idx];
                constexpr uint32_t n_words = mini_block_size / 8;
                uint32_t wi = in_block / 8;
                uint32_t skip = (in_block % 8) * 8 + bit_off;

                uint64_t w = words[wi] >> skip;
                if(w){
                    pos = (uint64_t)block_idx * mini_block_size * 8
                        + (uint64_t)wi * 64 + skip + ctz_u64(w);
                    current_pos = pos + 1;
                    return pos;
                }
                for(++wi; wi < n_words; ++wi){
                    if(words[wi]){
                        pos = (uint64_t)block_idx * mini_block_size * 8
                            + (uint64_t)wi * 64 + ctz_u64(words[wi]);
                        current_pos = pos + 1;
                        return pos;
                    }
                }
                in_block = 0;
                bit_off = 0;
            }
            current_pos = total_bits;
            return total_bits;
        }
        uint64_t get_next_prime_enc(){
            while(true){
                uint64_t np = get_next_pos();
                if(np < (upper_bound-lower_bound)*8)
                    return np;
                if(np >= (max_sieve-lower_bound*30))
                    return ~0ull; // failure mode
                // extend sieving code here ...
                sieve_next();
            }
        }
        uint64_t get_next_prime(){
            return pos_to_prime(get_next_prime_enc());
        }
        sieve_ctx(sieve_kernel* kernel);
    };

    struct sieve_kernel{
        block_vec<uint32_t, 1u<<18> primes;
        uint8_t subgroup[8][8];
        uint8_t subgroup_skip[8][8], subgroup_skip_mul[8][8];
        // {c | c coprime 30} forms a multiplicative subgroup of size 8
        // 30*N+c sieves the positions of exactly (30*N+c)*(30*M+c2)
        // thus, fixing c, all 30*N+c sieves in a similar pattern,
        // the sieving interval is only changed by a fixed delta
        // and it is exactly N
        // so there's no need for us to record different sieving
        // intervals for every prime > 30
        // the interval pattern of every c can be precomputed and
        // stored in a 8x8 byte matrix, very easy isn't it

        unique_ptr<sieve_ctx> small_primes_sieve;
        
        inline uint64_t subgroup_mul(uint64_t a, uint64_t b){
            // (30*N+c)*(30*M+d)=900*N*M+30*(N*d+M*c)+c*d
            // = (30*(a/8)*(b/8) + (a/8*pos[b&7] + b/8*pos[a&7]))*8 + subgroup[a&7][b&7]
            uint64_t ah = a/8, al = a&7;
            uint64_t bh = b/8, bl = b&7;
            return (30*ah*bh + ah*sieve_pos[bl] + bh*sieve_pos[al])*8 + (subgroup[al][bl]&7);
        }

        inline uint64_t subgroup_sqr(uint32_t a){
            uint64_t ah = a/8, al = a&7;
            return (30*ah + 2*sieve_pos[al])*ah*8 + (subgroup[al][al]&7);
        }

        sieve_kernel(){
            for(int i=0;i<8;++i)
                for(int j=0;j<8;++j){
                    int a = sieve_pos[i], b = sieve_pos[j], c = sieve_pos[j+1&7];
                    c = c<b?c+30:c;
                    subgroup_skip_mul[i][sieve_idx[b*a%30]]=c-b;
                    b*=a; c*=a;
                    subgroup[i][j] = (b/30)*8+sieve_idx[b%30];
                    subgroup_skip[i][sieve_idx[b%30]]=(c/30-b/30)*8+(sieve_idx[c%30]-sieve_idx[b%30]);
                }
            small_primes_sieve = make_unique<sieve_ctx>(this);
        }
        void add_prime(uint32_t prime){
            primes.push_back(prime);
        }
    };
    sieve_kernel* get_sieve_kernel(){
        static sieve_kernel kern;
        return &kern;
    }
    void sieve_intermediate(uint8_t* sieve,uint32_t* primes,uint32_t* offset,uint32_t n_primes, struct sieve_kernel* kern){
        auto skip_mat = kern->subgroup_skip;
        auto skip_mul = kern->subgroup_skip_mul;
        uint8_t masks[8];
        for(uint32_t i=0;i<8;++i) masks[i] = ~(1u<<i);
        for(uint32_t i=0;i<n_primes;++i){
            uint32_t N = primes[i]/8*8, c = primes[i]&7, j=offset[i];
            for(;j<mini_block_size*8;j+=skip_mat[c][j&7]+N*skip_mul[c][j&7])
                sieve[j/8]&=masks[j&7];
            offset[i]=j-mini_block_size*8;
        }
    }
    sieve_ctx::sieve_ctx(sieve_kernel* kernel){
        // a bootstrap
        lower_bound = 0u;
        upper_bound = mini_block_size;
        max_sieve = 143165577; // max_sieve * 30 > 2^32
        memset(small_offsets, 0, sizeof(small_offsets));
        // initial sieve: sieve 16384*30
        uint8_t* first_block = *sieve.no_init_extend();
        current_pos = 0;
        sieve_small(first_block, small_offsets);
        uint64_t prime;
        do{
            prime = get_next_prime_enc();
            kernel->add_prime(prime);
            // set up offset
            prime_offsets.push_back(kernel->subgroup_sqr(prime));
            prime = pos_to_prime(prime);
        }while(prime * prime <= mini_block_size*30);
        // do sieving
        sieve_intermediate(first_block, kernel->primes+0, prime_offsets+0, kernel->primes.size(), kernel);
    }
    // Now implement bucket sieve
    constexpr uint64_t large_prime_threshold = mini_block_size*4;
    // corresponds to 16384*15

}
}