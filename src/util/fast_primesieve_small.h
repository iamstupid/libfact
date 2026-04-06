#pragma once
// fast_primesieve_v2.h — Segmented sieve of Eratosthenes with mod-30 wheel
//
// Architecture:
//   prime_table (singleton)  — self-extending prime list, no circular deps
//   sieve_small / sieve_intermediate / bucket_sieve — free functions
//   prime_sieve (public API) — segmented window sieve for user queries
//
#include <cstdint>
#include <cstring>
#include <cmath>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
#include "intrin.h"
#include "popcnt.h"

namespace zfactor {
namespace sieve {
    using namespace std;

    // ================================================================
    //  Constants
    // ================================================================

    constexpr uint32_t mini_block = 16384;          // 16 KiB, L1-friendly
    constexpr uint32_t default_window = 32 * mini_block; // 512 KiB

    inline constexpr uint8_t sieve_pos[8] = {1,7,11,13,17,19,23,29};
    inline constexpr uint8_t sieve_idx[30] = {
        0xFF,0, 0xFF,0xFF,0xFF,0xFF,0xFF,1, 0xFF,0xFF,
        0xFF,2, 0xFF,3,    0xFF,0xFF,0xFF,4, 0xFF,5,
        0xFF,0xFF,0xFF,6, 0xFF,0xFF,0xFF,0xFF,0xFF,7,
    };
    
    // presieve primes: wheel {2,3,5} + pattern primes {7..227}
    inline constexpr uint8_t presieve_primes[] = {
        2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,
        61,67,71,73,79,83,89,97,101,103,107,109,113,
        127,131,137,139,149,151,157,163,167,173,179,181,
        191,193,197,199,211,223,227
    };

    constexpr const uint8_t* small_primes = presieve_primes + 3;

    constexpr uint32_t n_small_primes = size(presieve_primes) - 3; // - 2,3,5
    constexpr uint32_t n_small_fold = 6;
    constexpr uint32_t n_patterns = n_small_primes - n_small_fold/2;
    constexpr uint32_t n_presieve = sizeof(presieve_primes)/sizeof(presieve_primes[0]);

    // bucket sieve threshold: primes with stride > mini_block*2 use buckets
    // pi(mini_block*2*30) - n_small_primes - 3 ≈ 11505
    constexpr uint32_t lp_ind_start = 11505;

    // ================================================================
    //  Encoding helpers
    // ================================================================

    // encoded prime = (p/30)*8 + sieve_idx[p%30]
    // decode: (enc/8)*30 + sieve_pos[enc&7]
    inline uint64_t decode(uint32_t enc) {
        return (uint64_t)(enc/8)*30 + sieve_pos[enc&7];
    }

    inline uint64_t align_up(uint64_t value, uint64_t align) {
        return (value + align - 1) / align * align;
    }

    inline void harvest(uint8_t* sieve, uint32_t bytes,
        uint32_t base_enc, std::vector<uint32_t>& out)
    {
        for(uint32_t byte = 0; byte < bytes; ++byte) {
            uint8_t bits = sieve[byte];
            while(bits) {
                int bit = ctz_u64(bits);
                bits &= bits - 1;
                out.push_back(base_enc + byte*8 + bit);
            }
        }
    }

    // ================================================================
    //  Wheel tables
    // ================================================================

    struct wheel_data {
        uint8_t subgroup[8][8];
        uint8_t skip[8][8], skip_mul[8][8];

        wheel_data() {
            for(int i = 0; i < 8; ++i)
                for(int j = 0; j < 8; ++j) {
                    int a = sieve_pos[i], b = sieve_pos[j];
                    int c = sieve_pos[(j+1)&7];
                    c = c < b ? c+30 : c;
                    skip_mul[i][sieve_idx[b*a%30]] = c - b;
                    b *= a; c *= a;
                    subgroup[i][j] = (b/30)*8 + sieve_idx[b%30];
                    skip[i][sieve_idx[b%30]] = (c/30 - b/30)*8
                        + (sieve_idx[c%30] - sieve_idx[b%30]);
                }
        }

        uint64_t sqr(uint32_t enc) const {
            uint64_t ah = enc/8, al = enc&7;
            return (30*ah + 2*sieve_pos[al])*ah*8 + subgroup[al][al];
        }
    };

    inline uint32_t window_offset(uint64_t hit, uint64_t base_enc,
        uint32_t enc, const wheel_data& w)
    {
        if(hit < base_enc) {
            uint32_t N = enc/8*8, c = enc&7;
            uint64_t gap = base_enc - hit;
            uint64_t cycle = decode(enc) * 8;
            hit += (gap / cycle) * cycle;
            while(hit < base_enc)
                hit += w.skip[c][hit&7] + (uint64_t)N*w.skip_mul[c][hit&7];
        }
        return (uint32_t)(hit - base_enc);
    }

    // ================================================================
    //  Small prime pattern table
    // ================================================================

    constexpr uint32_t compute_pattern_size() {
        uint32_t total = 0, margin = vec_size - 1;
        for(uint32_t i = 0; i < n_small_fold-1-i; ++i)
            total += small_primes[i]*small_primes[n_small_fold-1-i] + margin;
        for(uint32_t i = n_small_fold; i < n_small_primes; ++i)
            total += small_primes[i] + margin;
        return total;
    }
    constexpr uint32_t pattern_size = compute_pattern_size();

    struct alignas(64) small_table {
        uint8_t pattern[pattern_size];
        uint8_t length[n_patterns], ilen[n_patterns];
        uint16_t offset[n_patterns];

        small_table() {
            constexpr uint32_t margin = vec_size - 1;
            memset(pattern, 0xFF, sizeof(pattern));
            offset[0] = 0;
            for(uint32_t i = 0; i < n_small_primes; ++i) {
                uint32_t pat_size = (i < n_small_fold
                    ? small_primes[i]*small_primes[n_small_fold-1-i]
                    : small_primes[i]);
                uint32_t max_pat = (pat_size + margin)*30;
                uint32_t pi = (i < n_small_fold
                    ? min(i, n_small_fold-1-i)
                    : i - n_small_fold/2);
                length[pi] = pat_size;
                ilen[pi] = vec_size % pat_size;
                if(i && (i < n_small_fold/2 || i >= n_small_fold))
                    offset[pi] = offset[pi-1] + length[pi-1] + margin;
                uint8_t* pat = pattern + offset[pi];
                for(uint32_t j = small_primes[i]; j < max_pat; j += small_primes[i]) {
                    uint32_t li = j/30;
                    uint8_t bi = sieve_idx[j%30];
                    if(bi != 0xFF) pat[li] &= ~(1u << bi);
                }
            }
        }
    };

    inline const small_table& get_small_table() {
        static const small_table st;
        return st;
    }

    // ================================================================
    //  Sieve stages (free functions)
    // ================================================================

    inline void sieve_small(uint8_t* sieve, uint8_t* offsets) {
        memset(sieve, 0xFF, mini_block);
        auto& st = get_small_table();
        for(uint32_t i = 0; i < n_patterns; ++i) {
            const uint8_t* pat = st.pattern + st.offset[i];
            uint16_t off = offsets[i];
            uint8_t len = st.length[i], delta = st.ilen[i];
            for(uint16_t j = 0; j < mini_block; j += vec_size) {
                (vec::load(sieve+j) & vec::load(pat+off)).store(sieve+j);
                off += delta;
                if(off >= len) off -= len;
            }
            offsets[i] = (uint8_t)off;
        }
    }

    inline void sieve_intermediate(uint8_t* block,
        const uint32_t* primes, uint32_t* offsets, uint32_t n,
        const wheel_data& w)
    {
        uint8_t masks[8];
        for(int i = 0; i < 8; ++i) masks[i] = ~(1u << i);
        for(uint32_t i = 0; i < n; ++i) {
            uint32_t N = primes[i]/8*8, c = primes[i]&7, j = offsets[i];
            for(; j < mini_block*8; j += w.skip[c][j&7] + N*w.skip_mul[c][j&7])
                block[j/8] &= masks[j&7];
            offsets[i] = j - mini_block*8;
        }
    }

    // ================================================================
    //  Bucket sieve
    // ================================================================

    struct bucket {
        static constexpr uint32_t capacity = 1023;
        uint16_t length;
        uint16_t data[capacity];
        void clear() { length = 0; }
        void push(uint16_t v) { data[length++] = v; }
        bool full() const { return length >= capacity; }
    };

    inline void bucket_sieve(uint8_t* sieve, uint32_t window_bytes,
        const uint32_t* primes, uint32_t* offsets,
        uint32_t p_start, uint32_t p_end,
        const wheel_data& w,
        bucket* buckets, uint32_t n_buckets)
    {
        uint32_t window_bits = window_bytes * 8;
        uint8_t masks[8];
        for(int i = 0; i < 8; ++i) masks[i] = ~(1u << i);

        uint32_t p_ind = p_start;
        while(p_ind < p_end) {
            for(uint32_t i = 0; i < n_buckets; ++i) buckets[i].clear();
            bool any_full = false;
            while(!any_full && p_ind < p_end) {
                uint32_t p = primes[p_ind];
                uint32_t N = p/8*8, c = p&7, j = offsets[p_ind];
                bool finished_prime = true;
                while(j < window_bits) {
                    bucket& bkt = buckets[j >> 16];
                    bkt.push((uint16_t)(j & 0xFFFF));
                    j += w.skip[c][j&7] + N*w.skip_mul[c][j&7];
                    if(bkt.full()) {
                        any_full = true;
                        finished_prime = false;
                        break;
                    }
                }
                if(finished_prime) {
                    offsets[p_ind] = j - window_bits;
                    ++p_ind;
                } else {
                    offsets[p_ind] = j;
                }
            }
            for(uint32_t bi = 0; bi < n_buckets; ++bi) {
                uint8_t* blk = sieve + bi*8192;
                auto& bkt = buckets[bi];
                for(uint16_t k = 0; k < bkt.length; ++k) {
                    uint16_t off = bkt.data[k];
                    blk[off >> 3] &= masks[off & 7];
                }
            }
        }
    }

    // Entry functions: require implementation
    void sieve(uint32_t lb, uint32_t ub, vector<uint32_t>& array_to_fill, uint32_t num_threads = 1);
    void sieve(uint64_t lb, uint64_t ub, vector<uint64_t>& array_to_fill, uint32_t num_threads = 1);

    // ================================================================
    //  prime_table — self-extending prime list (singleton)
    // ================================================================


    struct prime_table {
        wheel_data wheel;
        std::vector<uint32_t> primes;  // encoded, sorted, append-only
        uint64_t upper_bound = 0;      // bytes sieved so far (30-units)

        prime_table() {
            // bootstrap: sieve first block
            std::vector<uint8_t> buf_v(mini_block);
            uint8_t* buf = buf_v.data();
            uint8_t sm_off[n_patterns] = {};

            sieve_small(buf, sm_off);
            buf[0] &= 0xFE; // clear bit 0 (number 1)

            // The first block only needs primes up to sqrt(mini_block * 30),
            // so a one-shot intermediate sieve is enough here.
            uint64_t block_end = (uint64_t)mini_block * 30;
            std::vector<uint32_t> sp;
            std::vector<uint32_t> so;
            for(uint32_t byte = 0; byte < mini_block; ++byte) {
                uint8_t bits = buf[byte];
                while(bits) {
                    int b = ctz_u64(bits);
                    bits &= bits - 1;
                    uint32_t enc = byte*8 + b;
                    uint64_t p = decode(enc);
                    if(p*p > block_end) goto done_collect;
                    sp.push_back(enc);
                    so.push_back((uint32_t)wheel.sqr(enc));
                }
            }
            done_collect:

            sieve_intermediate(buf, sp.data(), so.data(),
                               (uint32_t)sp.size(), wheel);

            harvest(buf, mini_block, 0, primes);
            upper_bound = mini_block;
        }

        // ensure primes cover at least up to limit (in 30-units).
        // returns count of primes with encoded value < limit*8
        // (i.e. primes representing numbers < limit*30)
        uint32_t ensure_upto(uint32_t limit) {
            uint32_t enc_limit = limit * 8;
            if(primes.empty() || primes.back() < enc_limit) {
                uint64_t target = ((uint64_t)limit + mini_block - 1) / mini_block * mini_block;
                extend_to(target);
            }
            // binary search for first prime >= enc_limit
            auto it = std::lower_bound(primes.begin(), primes.end(), enc_limit);
            return (uint32_t)(it - primes.begin());
        }

    private:
        void extend_to(uint64_t target_ub) {
            if(target_ub <= upper_bound) return;

            std::vector<uint8_t> buf_v(mini_block);
            uint8_t* buf = buf_v.data();

            auto& st = get_small_table();
            uint8_t sm_off[n_patterns];
            for(uint32_t i = 0; i < n_patterns; ++i)
                sm_off[i] = (uint8_t)(upper_bound % st.length[i]);

            std::vector<uint32_t> sp, so;
            uint32_t sp_scan = 0;

            auto grow_sieve_primes = [&](uint64_t block_end_num) {
                while(sp_scan < primes.size()) {
                    uint32_t enc = primes[sp_scan];
                    uint64_t p = decode(enc);
                    if(p*p > block_end_num) break;
                    sp.push_back(enc);
                    so.push_back(window_offset(wheel.sqr(enc), upper_bound * 8, enc, wheel));
                    ++sp_scan;
                }
            };

            while(upper_bound < target_ub) {
                uint64_t block_end = (upper_bound + mini_block) * 30;
                grow_sieve_primes(block_end);
                sieve_small(buf, sm_off);
                sieve_intermediate(buf, sp.data(), so.data(),
                                   (uint32_t)sp.size(), wheel);
                harvest(buf, mini_block, (uint32_t)(upper_bound * 8), primes);
                upper_bound += mini_block;
            }
        }
    };

    inline prime_table& get_prime_table() {
        static prime_table pt;
        return pt;
    }

    // ================================================================
    //  Presieve helpers
    // ================================================================

    inline uint32_t count_presieve(uint64_t lo, uint64_t hi) {
        uint32_t c = 0;
        for(uint32_t i = 0; i < n_presieve; ++i) {
            if(presieve_primes[i] >= hi) break;
            if(presieve_primes[i] >= lo) ++c;
        }
        return c;
    }

    // ================================================================
    //  prime_sieve — public API
    // ================================================================

    struct prime_sieve {
        uint64_t lb, ub;              // user-requested [lb, ub)
        uint64_t lower_bound;         // current window start (30-units)
        uint64_t upper_bound;         // current window end (30-units)
        uint64_t max_sieve;           // = ub (number space)
        uint32_t required_np;         // primes used for sieving
        uint32_t window_size;         // bytes per window
        uint64_t current_pos;         // bit position within current window
        uint32_t presieve_idx;

        std::vector<uint8_t> sieve;
        std::vector<uint32_t> offsets;
        std::vector<bucket> buckets;
        uint8_t small_offsets[n_patterns];

        prime_sieve(uint64_t lower, uint64_t upper,
                    uint32_t win_size = default_window)
            : lb(lower), ub(upper),
              max_sieve(upper), window_size(win_size),
              required_np(0), current_pos(0), presieve_idx(0)
        {
            if(ub <= lb) { lower_bound = upper_bound = 0; return; }

            // snap lower_bound to mini_block boundary
            lower_bound = (max(lb, (uint64_t)229) / 30) / mini_block * mini_block;
            upper_bound = lower_bound;

            // advance presieve_idx past primes below lb
            while(presieve_idx < n_presieve && presieve_primes[presieve_idx] < lb)
                ++presieve_idx;

            // compute first window
            uint64_t max_bound = (max_sieve - 1) / 30 + 1;
            uint64_t first_ub = min(lower_bound + window_size, max_bound);
            first_ub = ((first_ub - 1) / mini_block + 1) * mini_block; // round up

            // pre-allocate
            sieve.resize(window_size);
            uint32_t n_buckets = window_size / 8192;
            buckets.resize(n_buckets);

            // sieve first window
            init_window(lower_bound, first_ub);

            // skip past primes below lb
            if(lower_bound * 30 < lb) {
                while(true) {
                    uint64_t np = get_next_pos();
                    if(np >= (upper_bound - lower_bound)*8) break;
                    if(pos_to_prime(np) >= lb) {
                        current_pos = np;
                        break;
                    }
                }
            }
        }

        // --- iteration ---

        uint64_t next_prime() {
            // presieve primes first
            if(presieve_idx < n_presieve) {
                uint64_t p = presieve_primes[presieve_idx];
                if(p < ub) { ++presieve_idx; return p; }
                presieve_idx = n_presieve;
            }
            if(ub <= 229) return 0;
            uint64_t enc = get_next_prime_enc();
            if(enc == ~0ull) return 0;
            uint64_t p = pos_to_prime(enc);
            return p < ub ? p : 0;
        }

        void collect(std::vector<uint64_t>& out) {
            for(uint64_t p; (p = next_prime()) != 0; )
                out.push_back(p);
        }

        // --- counting ---

        uint64_t count_primes() {
            uint64_t count = count_presieve(lb, ub);
            if(ub <= 229) return count;
            uint64_t scan_lo = max(lb, (uint64_t)229);
            count += count_window_bits(scan_lo, ub);
            while(upper_bound * 30 < ub && upper_bound * 30 < max_sieve) {
                next_window();
                count += count_window_bits(scan_lo, ub);
            }
            return count;
        }

        // --- statics ---

        static uint64_t pi(uint64_t n) {
            if(n <= 2) return 0;
            prime_sieve s(2, n);
            return s.count_primes();
        }

        static std::vector<uint64_t> range(uint64_t lo, uint64_t hi,
                                           uint32_t num_threads = 1) {
            std::vector<uint64_t> out;
            if(lo >= hi) return out;
            ::zfactor::sieve::sieve(lo, hi, out, num_threads);
            return out;
        }

        static uint64_t nth_prime(uint64_t n) {
            if(n == 0) return 0;
            if(n <= n_presieve) return presieve_primes[n-1];
            uint64_t est = (uint64_t)((double)n * (log((double)n) + log(log((double)n)) + 3));
            prime_sieve s(2, est + 1);
            uint64_t count = 0, p = 0;
            while(count < n) {
                p = s.next_prime();
                if(p == 0) break;
                ++count;
            }
            return p;
        }

    private:
        uint64_t pos_to_prime(uint64_t q) const {
            return (lower_bound + q/8)*30 + sieve_pos[q&7];
        }

        uint64_t get_next_pos() {
            uint32_t total = (uint32_t)(upper_bound - lower_bound);
            uint64_t total_bits = (uint64_t)total * 8;
            uint64_t pos = current_pos;
            if(pos >= total_bits) return total_bits;

            uint32_t byte_off = (uint32_t)(pos / 8);
            uint32_t bit_off = (uint32_t)(pos & 7);

            // align to word boundary
            uint32_t word_start = byte_off & ~7u;
            const uint64_t* words = (const uint64_t*)(sieve.data() + word_start);
            uint32_t n_words = (total - word_start) / 8;
            uint32_t wi = 0;
            uint32_t skip = (byte_off - word_start)*8 + bit_off;

            uint64_t w = words[wi] >> skip;
            if(w) {
                pos = ((uint64_t)word_start + (uint64_t)wi*8)*8 + skip + ctz_u64(w);
                current_pos = pos + 1;
                return pos;
            }
            for(++wi; wi < n_words; ++wi) {
                if(words[wi]) {
                    pos = ((uint64_t)word_start + (uint64_t)wi*8)*8 + ctz_u64(words[wi]);
                    current_pos = pos + 1;
                    return pos;
                }
            }
            current_pos = total_bits;
            return total_bits;
        }

        uint64_t get_next_prime_enc() {
            while(true) {
                uint64_t np = get_next_pos();
                if(np < (upper_bound - lower_bound)*8)
                    return np;
                if(upper_bound * 30 >= max_sieve)
                    return ~0ull;
                next_window();
            }
        }

        void init_window(uint64_t wlb, uint64_t wub) {
            auto& pt = get_prime_table();
            uint32_t sqrtub = isqrt(wub * 30) / 30 + 1;
            uint32_t old_np = required_np;
            required_np = pt.ensure_upto(sqrtub);

            // resize offsets if needed
            if(offsets.size() < required_np)
                offsets.resize(required_np);

            // compute offsets for new primes
            uint64_t lb_enc = wlb * 8;
            for(uint32_t i = old_np; i < required_np; ++i) {
                uint32_t p = pt.primes[i];
                offsets[i] = window_offset(pt.wheel.sqr(p), lb_enc, p, pt.wheel);
            }

            // set window state
            lower_bound = wlb;
            upper_bound = wub;
            current_pos = 0;
            uint32_t window_bytes = (uint32_t)(wub - wlb);

            // compute small_offsets
            auto& st = get_small_table();
            for(uint32_t i = 0; i < n_patterns; ++i)
                small_offsets[i] = (uint8_t)(wlb % st.length[i]);

            // sieve: small + intermediate per mini-block
            uint32_t n_inter = min(required_np, lp_ind_start);
            for(uint32_t b = 0; b < window_bytes; b += mini_block) {
                uint8_t* blk = sieve.data() + b;
                sieve_small(blk, small_offsets);
                sieve_intermediate(blk, pt.primes.data(), offsets.data(),
                                   n_inter, pt.wheel);
            }

            // bucket sieve for large primes
            if(required_np > lp_ind_start) {
                uint32_t n_bkt = window_bytes / 8192;
                if(buckets.size() < n_bkt) buckets.resize(n_bkt);
                bucket_sieve(sieve.data(), window_bytes,
                             pt.primes.data(), offsets.data(),
                             lp_ind_start, required_np,
                             pt.wheel, buckets.data(), n_bkt);
            }

            if(wlb == 0)
                sieve[0] &= 0xFE;
        }

        void next_window() {
            uint64_t new_lb = upper_bound;
            uint64_t max_bound = (max_sieve - 1) / 30 + 1;
            uint64_t new_ub = min(new_lb + window_size, max_bound);
            new_ub = ((new_ub - 1) / mini_block + 1) * mini_block; // round up
            if(new_ub <= new_lb) new_ub = new_lb + mini_block;
            init_window(new_lb, new_ub);
        }

        // popcount bits in [num_lo, num_hi) within current window
        uint64_t count_window_bits(uint64_t num_lo, uint64_t num_hi) const {
            uint64_t sieve_base = lower_bound * 30;
            uint64_t sieve_end  = upper_bound * 30;
            uint64_t eff_lo = max(num_lo, sieve_base);
            uint64_t eff_hi = min(num_hi, sieve_end);
            if(eff_lo >= eff_hi) return 0;

            uint32_t sieve_bytes = (uint32_t)(upper_bound - lower_bound);
            uint32_t first_byte = (uint32_t)((eff_lo - sieve_base) / 30);
            uint32_t last_byte  = (uint32_t)((eff_hi - sieve_base) / 30);
            if(last_byte > sieve_bytes) last_byte = sieve_bytes;

            // first byte mask
            uint8_t first_mask = 0xFF;
            {
                uint64_t base = sieve_base + (uint64_t)first_byte * 30;
                for(uint8_t b = 0; b < 8; ++b)
                    if(base + sieve_pos[b] < eff_lo)
                        first_mask &= ~(1u << b);
            }
            // last byte mask
            uint8_t last_mask = 0x00;
            if(last_byte < sieve_bytes) {
                uint64_t base = sieve_base + (uint64_t)last_byte * 30;
                for(uint8_t b = 0; b < 8; ++b)
                    if(base + sieve_pos[b] < eff_hi)
                        last_mask |= (1u << b);
            }

            if(first_byte == last_byte)
                return popcount_u64(sieve[first_byte] & first_mask & last_mask);
            if(first_byte > last_byte) return 0;

            uint64_t count = popcount_u64(sieve[first_byte] & first_mask);

            // bulk popcount middle bytes
            uint32_t mid_start = first_byte + 1, mid_end = last_byte;
            if(mid_start < mid_end) {
                const uint64_t* words = (const uint64_t*)(sieve.data() + mid_start);
                uint32_t n_bytes = mid_end - mid_start;
                count += popcount_array(words, n_bytes / 8);
                uint32_t tail = n_bytes & 7;
                if(tail) {
                    uint64_t last = 0;
                    memcpy(&last, sieve.data() + mid_end - tail, tail);
                    count += popcount_u64(last);
                }
            }

            if(last_mask && last_byte < sieve_bytes)
                count += popcount_u64(sieve[last_byte] & last_mask);

            return count;
        }
    };

    constexpr uint64_t parallel_grain = (uint64_t)default_window * 30 * 2;

    inline uint32_t choose_threads(uint64_t span, uint32_t requested) {
        if(requested == 0) {
            requested = std::thread::hardware_concurrency();
            if(requested == 0) requested = 1;
        }
        if(requested <= 1 || span <= parallel_grain) return 1;
        uint64_t by_work = (span + parallel_grain - 1) / parallel_grain;
        return (uint32_t)min<uint64_t>(requested, max<uint64_t>(1, by_work));
    }

    inline void prewarm_prime_table(uint64_t ub) {
        if(ub <= 229) return;
        get_prime_table().ensure_upto(isqrt(ub - 1) / 30 + 1);
    }

    template<class UInt>
    inline void sieve_serial(uint64_t lb, uint64_t ub, vector<UInt>& out) {
        out.clear();
        if(ub <= lb) return;
        prime_sieve ps(lb, ub);
        for(uint64_t p; (p = ps.next_prime()) != 0; )
            out.push_back((UInt)p);
    }

    template<class UInt>
    inline void sieve_parallel(uint64_t lb, uint64_t ub,
        vector<UInt>& out, uint32_t num_threads)
    {
        out.clear();
        if(ub <= lb) return;
        uint64_t span = ub - lb;
        uint32_t threads = choose_threads(span, num_threads);
        if(threads == 1) {
            sieve_serial(lb, ub, out);
            return;
        }

        prewarm_prime_table(ub);

        uint64_t chunk = align_up((span + threads - 1) / threads,
                                  (uint64_t)mini_block * 30);
        std::vector<std::vector<UInt>> partials(threads);
        std::vector<std::thread> workers;
        workers.reserve(threads);
        std::exception_ptr error;
        std::mutex error_lock;

        for(uint32_t i = 0; i < threads; ++i) {
            uint64_t lo = lb + (uint64_t)i * chunk;
            uint64_t hi = min(ub, lo + chunk);
            if(lo >= hi) break;
            workers.emplace_back([&, i, lo, hi]() {
                try {
                    sieve_serial(lo, hi, partials[i]);
                } catch(...) {
                    std::lock_guard<std::mutex> guard(error_lock);
                    if(!error) error = std::current_exception();
                }
            });
        }

        for(auto& worker : workers)
            worker.join();
        if(error) std::rethrow_exception(error);

        size_t total = 0;
        for(auto& part : partials) total += part.size();
        out.clear();
        out.reserve(total);
        for(auto& part : partials)
            out.insert(out.end(), part.begin(), part.end());
    }

    inline void sieve(uint32_t lb, uint32_t ub,
        vector<uint32_t>& array_to_fill, uint32_t num_threads)
    {
        sieve_parallel((uint64_t)lb, (uint64_t)ub, array_to_fill, num_threads);
    }

    inline void sieve(uint64_t lb, uint64_t ub,
        vector<uint64_t>& array_to_fill, uint32_t num_threads)
    {
        sieve_parallel(lb, ub, array_to_fill, num_threads);
    }

} // namespace sieve

// re-export for compatibility
using prime_sieve = sieve::prime_sieve;

} // namespace zfactor
