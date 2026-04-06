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
#include "util/intrin.h"

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
        block_vec<uint8_t, mini_block_size * mini_per_alloc> sieve;
        uint64_t lower_bound, upper_bound, max_sieve, current_pos;
        uint32_t required_np;
        bool bootstrap_mode = false; // breaks circular ensure_upto dependency
        // max_sieve is the sieving upbound
        // lower_bound and upper_bound are the bounds of current sieving window
        // each * 30 is the actual bound. i.e. [lower_bound*30, upper_bound*30)
        // required_np is the max prime needed to sieve to upper_bound
        // i.e. pi(isqrt(upper_bound*30)) - #small_primes
        uint8_t small_offsets[n_patterns];
        block_vec<uint32_t, 1u<<18> prime_offsets;
        // prime starting offset relative to lower_bound
        struct bucket{ // make a bucket 4KiB
            uint16_t length, bucks[n_bucklen];
            void clear(){ length = 0; }
            void push_back(uint16_t t){ bucks[length++] = t; }
            bool is_full(){ return length >= n_bucklen; }
        };
        block_vec<bucket, mini_per_alloc << 3> buckets;
        void initialize_window(uint64_t lb, uint64_t ub);
        void sieve_next();
        void bucket_stage();
        void initialize_next_window(){
            uint64_t new_lb = upper_bound;
            uint32_t isqrt_lb = isqrt(new_lb*30);
            uint32_t block_needed = (isqrt_lb-1)/(mini_block_size*30)+1;
            uint64_t new_ub = new_lb + (uint64_t)block_needed * mini_block_size;
            uint64_t max_bound = ((max_sieve-1)/(30*mini_block_size)+1)*mini_block_size;
            if(new_ub > max_bound) new_ub = max_bound;
            if(new_lb + sieve.capacity() >= max_bound){
                // if max is well within allocated block, then do it at once
                new_ub = max_bound;
            }
            if(new_ub <= new_lb) new_ub = new_lb + mini_block_size;
            initialize_window(new_lb, new_ub);
        }
        inline uint64_t pos_to_prime(uint64_t q){
            return (lower_bound + q/8)*30 + sieve_pos[q & 7];
        }
        static constexpr uint32_t sieve_bpa = mini_block_size * mini_per_alloc;
        uint64_t get_next_pos(){
            uint64_t total_bits = (uint64_t)sieve.size() * 8;
            uint64_t pos = current_pos;
            if(pos >= total_bits) return total_bits;

            uint32_t byte_off = (uint32_t)(pos / 8);
            uint32_t block_idx = byte_off / sieve_bpa;
            uint32_t in_block = byte_off % sieve_bpa;
            uint32_t bit_off = (uint32_t)(pos & 7);

            for(; block_idx < sieve.blocks.size(); ++block_idx){
                const uint64_t* words = (const uint64_t*)sieve.block_ptr(block_idx * sieve_bpa);
                uint32_t block_bytes = sieve.block_count(block_idx * sieve_bpa);
                uint32_t n_words = block_bytes / 8;
                uint32_t wi = in_block / 8;
                uint32_t skip = (in_block % 8) * 8 + bit_off;

                uint64_t w = words[wi] >> skip;
                if(w){
                    pos = ((uint64_t)block_idx * sieve_bpa + (uint64_t)wi * 8) * 8
                        + skip + ctz_u64(w);
                    current_pos = pos + 1;
                    return pos;
                }
                for(++wi; wi < n_words; ++wi){
                    if(words[wi]){
                        pos = ((uint64_t)block_idx * sieve_bpa + (uint64_t)wi * 8) * 8
                            + ctz_u64(words[wi]);
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
                // exhausted current window — are we done?
                if(upper_bound * 30 >= max_sieve)
                    return ~0ull;
                sieve_next();
            }
        }
        uint64_t get_next_prime(){
            return pos_to_prime(get_next_prime_enc());
        }
        sieve_ctx(uint64_t lb, uint64_t ub);
    private:
        friend struct sieve_kernel;
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
            return (30*ah*bh + ah*sieve_pos[bl] + bh*sieve_pos[al])*8 + subgroup[al][bl];
        }

        inline uint64_t subgroup_sqr(uint32_t a){
            uint64_t ah = a/8, al = a&7;
            return (30*ah + 2*sieve_pos[al])*ah*8 + subgroup[al][al];
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
            small_primes_sieve.reset(new sieve_ctx(this));
        }
        void add_prime(uint32_t prime){
            primes.push_back(prime);
        }
        // fetch next prime from bootstrap sieve
        uint32_t fetch_abs_enc(){
            uint64_t enc = small_primes_sieve->get_next_prime_enc();
            return (uint32_t)(enc + small_primes_sieve->lower_bound * 8);
        }
        uint32_t ensure_fetch(uint32_t nth){
            while(primes.size() <= nth)
                primes.push_back(fetch_abs_enc());
            return primes[nth];
        }
        // point is in 30-units; primes are in encoded space (30-units * 8 + wheel)
        uint32_t ensure_upto(uint32_t point){
            uint32_t enc_point = point * 8;
            while(primes.back() < enc_point)
                primes.push_back(fetch_abs_enc());
            return primes.size()-1;
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
        bootstrap_mode = true;
        lower_bound = 0u;
        upper_bound = mini_block_size;
        max_sieve = (uint64_t)143165577 * 30; // > 2^32, covers all uint32_t primes
        memset(small_offsets, 0, sizeof(small_offsets));
        // initial sieve: sieve 16384*30
        uint32_t base = sieve.size();
        sieve.resize(base + mini_block_size);
        uint8_t* first_block = sieve + base;
        current_pos = 1; // skip position 0 which represents number 1 (not prime)
        sieve_small(first_block, small_offsets);
        first_block[0] &= 0xFE; // clear bit 0 (number 1)
        uint64_t prime;
        do{
            prime = get_next_prime_enc();
            kernel->primes.push_back((uint32_t)prime);
            // set up offset
            prime_offsets.push_back(kernel->subgroup_sqr((uint32_t)prime));
            prime = pos_to_prime(prime);
        }while(prime * prime <= mini_block_size*30);
        required_np = kernel->primes.size();
        // do sieving
        sieve_intermediate(first_block, kernel->primes+0, prime_offsets+0, required_np, kernel);
    }
    // bucket sieve constants — must precede functions that reference them
    constexpr uint64_t large_prime_threshold = mini_block_size*2;
    constexpr uint32_t n_pi_lp = 11554;
    constexpr uint32_t lp_ind_start = n_pi_lp - size(small_primes) - 3;
    // corresponds to 8192*15 -- We HAVE to use 8192 block for bucket sieve
    // since 8192*8=65536 (uint16_t)

    // sieve the entire current window (small + intermediate + bucket)
    inline void sieve_window(sieve_ctx& ctx){
        auto kernel = get_sieve_kernel();
        uint32_t window_bytes = (uint32_t)(ctx.upper_bound - ctx.lower_bound);
        uint32_t n_inter = min(ctx.required_np, lp_ind_start);
        for(uint32_t b = 0; b < window_bytes; b += mini_block_size){
            uint8_t* blk = ctx.sieve + b;
            sieve_small(blk, ctx.small_offsets);
            sieve_intermediate(blk, kernel->primes + 0, ctx.prime_offsets + 0, n_inter, kernel);
        }
        ctx.bucket_stage();
    }

    void sieve_ctx::sieve_next(){
        if(bootstrap_mode){
            // The bootstrap sieve must not call initialize_window (which calls
            // ensure_upto -> fetch_abs_enc -> back into this sieve = infinite recursion).
            // Instead, extend one block at a time with the primes already available.
            sieve.resize(sieve.size() + mini_block_size);
            uint8_t* blk = sieve + (upper_bound - lower_bound);
            upper_bound += mini_block_size;
            sieve_small(blk, small_offsets);
            sieve_intermediate(blk, get_sieve_kernel()->primes + 0,
                               prime_offsets + 0, required_np, get_sieve_kernel());
            return;
        }
        initialize_next_window();
        sieve_window(*this);
    }
    sieve_ctx::sieve_ctx(uint64_t lb, uint64_t ub){
        // snap lower_bound down to mini_block_size boundary (in 30-units)
        lower_bound = (lb / 30) / mini_block_size * mini_block_size;
        // max_sieve is the literal upper bound in number space
        max_sieve = ub;
        required_np = 0;
        memset(small_offsets, 0, sizeof(small_offsets));
        current_pos = 0;
        // advance small_offsets to match lower_bound
        if(lower_bound > 0){
            const small_table* st = get_small_table();
            for(uint32_t i = 0; i < n_patterns; ++i){
                uint8_t len = st->length[i];
                small_offsets[i] = (uint8_t)((uint64_t)lower_bound % len);
            }
        }
        // compute first window size: enough blocks to cover sqrt(ub)/30
        uint32_t isqrt_ub = isqrt(ub);
        uint32_t first_window_blocks = isqrt_ub / (mini_block_size * 30) + 1;
        uint64_t first_ub = lower_bound + (uint64_t)first_window_blocks * mini_block_size;
        // clamp: if everything fits, just go to the end
        uint64_t max_bound = (max_sieve - 1) / 30 + 1;
        if(first_ub >= max_bound)
            first_ub = ((max_bound - 1) / mini_block_size + 1) * mini_block_size;
        // initialize_window expects upper_bound == lower_bound for a fresh start
        upper_bound = lower_bound;
        initialize_window(lower_bound, first_ub);
        sieve_window(*this);
        // skip past primes below the requested lower bound
        if(lower_bound * 30 < lb){
            while(true){
                uint64_t np = get_next_pos();
                if(np >= (upper_bound - lower_bound) * 8) break;
                if(pos_to_prime(np) >= lb){
                    // found the first prime >= lb; rewind so next call returns it
                    current_pos = np;
                    break;
                }
            }
        }
    }
    void sieve_ctx::bucket_stage(){
        auto kernel = get_sieve_kernel();
        auto skip_mat = kernel->subgroup_skip;
        auto skip_mul = kernel->subgroup_skip_mul;
        uint32_t window_bits = (uint32_t)(upper_bound - lower_bound) * 8;
        uint32_t n_buckets = (uint32_t)(upper_bound - lower_bound) / 8192;
        uint8_t masks[8];
        for(uint32_t i = 0; i < 8; ++i) masks[i] = ~(1u << i);

        // 0. set p_ind to lp_ind_start
        uint32_t p_ind = lp_ind_start;
        // 1. loop while p_ind < required_np
        while(p_ind < required_np){
            // a. clear buckets
            for(uint32_t i = 0; i < n_buckets; ++i)
                buckets[i].clear();
            // b. gather into buckets
            bool any_full = false;
            while(!any_full && p_ind < required_np){
                uint32_t p = kernel->primes[p_ind];
                uint32_t N = p/8*8, c = p&7;
                uint32_t j = prime_offsets[p_ind];
                // u. march offset, scatter into buckets
                while(j < window_bits){
                    buckets[j >> 16].push_back((uint16_t)(j & 0xFFFF));
                    if(buckets[j >> 16].is_full()) any_full = true;
                    j += skip_mat[c][j&7] + N * skip_mul[c][j&7];
                }
                // v. offset -= window length (setup for next window)
                prime_offsets[p_ind] = j - window_bits;
                // w. increase p_ind
                ++p_ind;
            }
            // c. sieve from gathered buckets
            for(uint32_t bi = 0; bi < n_buckets; ++bi){
                uint8_t* blk = sieve + (bi * 8192);
                auto& bkt = buckets[bi];
                for(uint16_t k = 0; k < bkt.length; ++k){
                    uint16_t off = bkt.bucks[k];
                    blk[off >> 3] &= masks[off & 7];
                }
            }
        }
    }
    void sieve_ctx::initialize_window(uint64_t lb, uint64_t ub){
        auto kernel = get_sieve_kernel();
        uint32_t sqrtub = isqrt(ub * 30)/30 + 1;
        uint32_t old_np = required_np;
        required_np = kernel->ensure_upto(sqrtub);
        auto skip_mat = kernel->subgroup_skip;
        auto skip_mul = kernel->subgroup_skip_mul;

        // initialize offsets for new primes
        prime_offsets.resize(required_np);
        for(uint32_t i = old_np; i < required_np; ++i){
            uint32_t p = kernel->primes[i];
            uint32_t N = p/8*8, c = p&7;
            // start at p*p (encoded position relative to 0)
            uint64_t j = kernel->subgroup_sqr(p);
            // if p*p < lb*30, advance to the first hit >= lb
            uint64_t lb_enc = lb * 8;
            if(j < lb_enc){
                uint64_t prime_val = (uint64_t)N/8 * 30 + sieve_pos[c];
                uint64_t gap = lb_enc - j;
                uint64_t cycle_enc = prime_val * 8;
                j += (gap / cycle_enc) * cycle_enc;
                while(j < lb_enc)
                    j += skip_mat[c][j&7] + (uint64_t)N * skip_mul[c][j&7];
            }
            prime_offsets[i] = (uint32_t)(j - lb_enc);
        }

        // initialize buffers
        lower_bound = lb;
        upper_bound = ub;
        current_pos = 0;
        uint32_t window_bytes = (uint32_t)(ub - lb);
        sieve.resize(window_bytes);
        uint32_t n_buckets = window_bytes / 8192;
        buckets.resize(n_buckets);
        /* Not needed because we would clear buckets in bucket stage
        for(uint32_t i = 0; i < n_buckets; ++i)
            buckets[i].clear();
            */
    }
} // namespace fast_sieve

    // all primes not represented in the sieve bitmap:
    // wheel primes {2,3,5} + pattern-sieved small primes {7..227}
    static constexpr uint8_t presieve_primes[] = {
        2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,
        61,67,71,73,79,83,89,97,101,103,107,109,113,
        127,131,137,139,149,151,157,163,167,173,179,181,
        191,193,197,199,211,223,227
    };
    static constexpr uint32_t n_presieve = sizeof(presieve_primes) / sizeof(presieve_primes[0]);

    // count presieve primes in [lo, hi)
    inline uint32_t count_presieve(uint64_t lo, uint64_t hi) {
        uint32_t c = 0;
        for(uint32_t i = 0; i < n_presieve; ++i){
            uint64_t p = presieve_primes[i];
            if(p >= hi) break;
            if(p >= lo) ++c;
        }
        return c;
    }

    // SIMD popcount over a byte range inside a block_vec, block-aware
    inline uint64_t popcount_sieve_range(
        const block_vec<uint8_t, fast_sieve::mini_block_size * fast_sieve::mini_per_alloc>& sv,
        uint32_t byte_start, uint32_t byte_end)
    {
        constexpr uint32_t bpa = fast_sieve::sieve_ctx::sieve_bpa;
        uint64_t total = 0;
        uint32_t off = byte_start;
        uint32_t rem = byte_end - byte_start;
        while(rem > 0){
            uint32_t in_blk = off % bpa;
            uint32_t chunk = std::min(rem, bpa - in_blk);
            const uint8_t* base = (const uint8_t*)sv.block_ptr(off);
            const uint64_t* words = (const uint64_t*)(base + in_blk);
            total += popcount_array(words, chunk / 8);
            uint32_t tail = chunk & 7;
            if(tail){
                uint64_t last = 0;
                memcpy(&last, (const uint8_t*)words + (chunk & ~7u), tail);
                total += popcount_u64(last);
            }
            off += chunk;
            rem -= chunk;
        }
        return total;
    }

    // friendly wrapper around the raw sieve_ctx
    struct prime_sieve {
        fast_sieve::sieve_ctx ctx;
        uint64_t lb, ub;        // user-requested bounds [lb, ub)
        uint32_t presieve_idx;  // index into presieve_primes for iteration

        // sieve primes in [lower, upper)
        // if upper <= 229, only presieve primes matter; give ctx a minimal valid range
        prime_sieve(uint64_t lower, uint64_t upper)
            : ctx(std::max(lower, (uint64_t)229), std::max(upper, (uint64_t)230)),
              lb(lower), ub(upper), presieve_idx(0)
        {
            // advance presieve_idx past primes below lb
            while(presieve_idx < n_presieve && presieve_primes[presieve_idx] < lb)
                ++presieve_idx;
        }

        // --- iteration ---

        // return the next prime in [lb, ub), or 0 if exhausted
        uint64_t next_prime() {
            // first yield presieve primes in range
            if(presieve_idx < n_presieve){
                uint64_t p = presieve_primes[presieve_idx];
                if(p < ub){
                    ++presieve_idx;
                    return p;
                }
                presieve_idx = n_presieve; // done with presieve
            }
            // then yield from sieve
            if(ub <= 228) return 0;
            uint64_t enc = ctx.get_next_prime_enc();
            if(enc == ~0ull) return 0;
            uint64_t p = ctx.pos_to_prime(enc);
            return p < ub ? p : 0;
        }

        // collect all remaining primes into a vector
        void collect(std::vector<uint64_t>& out) {
            for(uint64_t p; (p = next_prime()) != 0; )
                out.push_back(p);
        }

        // --- counting via SIMD popcount ---

        // count sieve bits in [num_lo, num_hi) within the current window
        static uint64_t count_window_bits(
            const fast_sieve::sieve_ctx& c, uint64_t num_lo, uint64_t num_hi)
        {
            uint64_t sieve_base = c.lower_bound * 30;
            uint64_t sieve_end  = c.upper_bound * 30;
            uint64_t eff_lo = std::max(num_lo, sieve_base);
            uint64_t eff_hi = std::min(num_hi, sieve_end);
            if(eff_lo >= eff_hi) return 0;

            uint32_t sieve_bytes = c.sieve.size();
            uint32_t first_byte = (uint32_t)((eff_lo - sieve_base) / 30);
            uint32_t last_byte  = (uint32_t)((eff_hi - sieve_base) / 30);
            if(last_byte > sieve_bytes) last_byte = sieve_bytes;

            uint8_t first_mask = 0xFF;
            {
                uint64_t base = sieve_base + (uint64_t)first_byte * 30;
                for(uint8_t b = 0; b < 8; ++b)
                    if(base + fast_sieve::sieve_pos[b] < eff_lo)
                        first_mask &= ~(1u << b);
            }
            uint8_t last_mask = 0x00;
            if(last_byte < sieve_bytes){
                uint64_t base = sieve_base + (uint64_t)last_byte * 30;
                for(uint8_t b = 0; b < 8; ++b)
                    if(base + fast_sieve::sieve_pos[b] < eff_hi)
                        last_mask |= (1u << b);
            }

            if(first_byte == last_byte)
                return popcount_u64(c.sieve[first_byte] & first_mask & last_mask);
            if(first_byte > last_byte) return 0;

            uint64_t count = popcount_u64(c.sieve[first_byte] & first_mask);
            uint32_t full_start = first_byte + 1;
            uint32_t full_end = last_byte;
            if(full_start < full_end)
                count += popcount_sieve_range(c.sieve, full_start, full_end);
            if(last_mask && last_byte < sieve_bytes)
                count += popcount_u64(c.sieve[last_byte] & last_mask);
            return count;
        }

        // count primes in [lb, ub), sieving through all necessary windows
        uint64_t count_primes() {
            uint64_t count = count_presieve(lb, ub);
            if(ub <= 229) return count;

            // count the first (already-sieved) window
            uint64_t scan_lo = std::max(lb, (uint64_t)229);
            count += count_window_bits(ctx, scan_lo, ub);

            // sieve additional windows until we cover ub
            while(ctx.upper_bound * 30 < ub){
                if(ctx.upper_bound * 30 >= ctx.max_sieve) break;
                ctx.sieve_next();
                count += count_window_bits(ctx, scan_lo, ub);
            }
            return count;
        }

        // --- convenience statics ---

        // pi(n): count primes < n
        static uint64_t pi(uint64_t n) {
            if(n <= 2) return 0;
            prime_sieve s(2, n);
            return s.count_primes();
        }

        // collect all primes in [lower, upper)
        static std::vector<uint64_t> range(uint64_t lower, uint64_t upper) {
            std::vector<uint64_t> out;
            if(lower >= upper) return out;
            prime_sieve s(lower, upper);
            s.collect(out);
            return out;
        }

        // get the nth prime (1-indexed: nth_prime(1) = 2)
        static uint64_t nth_prime(uint64_t n) {
            if(n == 0) return 0;
            if(n <= n_presieve) return presieve_primes[n - 1];
            // upper bound: p_n < n*(ln(n) + ln(ln(n))) + 3
            uint64_t est = (uint64_t)((double)n * (log((double)n) + log(log((double)n)) + 3));
            prime_sieve s(2, est + 1);
            uint64_t count = 0, p = 0;
            while(count < n){
                p = s.next_prime();
                if(p == 0) break;
                ++count;
            }
            return p;
        }
    };

} // namespace zfactor