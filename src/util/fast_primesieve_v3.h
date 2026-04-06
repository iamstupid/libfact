#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include "intrin.h"
#include "popcnt.h"

namespace zfactor {
namespace sieve_v3 {

constexpr uint32_t mini_block = 16384;
constexpr uint32_t default_window_bytes = 512 * 1024;
constexpr uint32_t default_target_thread_bytes = 1u << 18;
constexpr uint32_t default_dense_prime = mini_block * 2 * 30;

inline constexpr uint8_t sieve_pos[8] = {1, 7, 11, 13, 17, 19, 23, 29};
inline constexpr uint8_t sieve_idx[30] = {
    0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1,    0xFF, 0xFF,
    0xFF, 2,    0xFF, 3,    0xFF, 0xFF, 0xFF, 4,    0xFF, 5,
    0xFF, 0xFF, 0xFF, 6,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 7,
};

inline constexpr uint8_t presieve_primes[] = {
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,
    61,67,71,73,79,83,89,97,101,103,107,109,113,
    127,131,137,139,149,151,157,163,167,173,179,181,
    191,193,197,199,211,223,227
};

constexpr const uint8_t* small_primes = presieve_primes + 3;
constexpr uint32_t n_small_primes = sizeof(presieve_primes) - 3;
constexpr uint32_t n_small_fold = 6;
constexpr uint32_t n_patterns = n_small_primes - n_small_fold / 2;
constexpr uint32_t n_presieve = sizeof(presieve_primes) / sizeof(presieve_primes[0]);

inline uint64_t decode(uint32_t enc) {
    return (uint64_t)(enc / 8) * 30 + sieve_pos[enc & 7];
}

inline uint64_t align_up(uint64_t value, uint64_t align) {
    return (value + align - 1) / align * align;
}

inline uint64_t div_ceil(uint64_t num, uint64_t den) {
    return (num + den - 1) / den;
}

inline uint32_t count_presieve(uint64_t lo, uint64_t hi) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < n_presieve; ++i) {
        uint64_t p = presieve_primes[i];
        if (p >= hi) break;
        if (p >= lo) ++count;
    }
    return count;
}

struct wheel_data {
    uint8_t subgroup[8][8];
    uint8_t skip[8][8];
    uint8_t skip_mul[8][8];

    wheel_data() {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                int a = sieve_pos[i];
                int b = sieve_pos[j];
                int c = sieve_pos[(j + 1) & 7];
                c = c < b ? c + 30 : c;
                skip_mul[i][sieve_idx[(b * a) % 30]] = (uint8_t)(c - b);
                b *= a;
                c *= a;
                subgroup[i][j] = (uint8_t)((b / 30) * 8 + sieve_idx[b % 30]);
                skip[i][sieve_idx[b % 30]] =
                    (uint8_t)((c / 30 - b / 30) * 8 + (sieve_idx[c % 30] - sieve_idx[b % 30]));
            }
        }
    }

    uint64_t sqr(uint32_t enc) const {
        uint64_t ah = enc / 8;
        uint64_t al = enc & 7;
        return (30 * ah + 2 * sieve_pos[al]) * ah * 8 + subgroup[al][al];
    }
};

inline uint64_t advance_hit(uint64_t hit, uint32_t enc, const wheel_data& wheel) {
    uint32_t N = enc / 8 * 8;
    uint32_t c = enc & 7;
    return hit + wheel.skip[c][hit & 7] + (uint64_t)N * wheel.skip_mul[c][hit & 7];
}

inline uint64_t advance_to_at_least(uint64_t hit, uint64_t base_enc,
                                    uint32_t enc, const wheel_data& wheel) {
    if (hit < base_enc) {
        uint64_t gap = base_enc - hit;
        uint64_t cycle = decode(enc) * 8;
        hit += (gap / cycle) * cycle;
        while (hit < base_enc)
            hit = advance_hit(hit, enc, wheel);
    }
    return hit;
}

constexpr uint32_t compute_pattern_size() {
    uint32_t total = 0;
    constexpr uint32_t margin = vec_size - 1;
    for (uint32_t i = 0; i < n_small_fold - 1 - i; ++i)
        total += small_primes[i] * small_primes[n_small_fold - 1 - i] + margin;
    for (uint32_t i = n_small_fold; i < n_small_primes; ++i)
        total += small_primes[i] + margin;
    return total;
}

constexpr uint32_t pattern_size = compute_pattern_size();

struct alignas(64) small_table {
    uint8_t pattern[pattern_size];
    uint8_t length[n_patterns];
    uint8_t ilen[n_patterns];
    uint16_t offset[n_patterns];

    small_table() {
        constexpr uint32_t margin = vec_size - 1;
        std::memset(pattern, 0xFF, sizeof(pattern));
        offset[0] = 0;
        for (uint32_t i = 0; i < n_small_primes; ++i) {
            uint32_t pat_size = i < n_small_fold
                ? small_primes[i] * small_primes[n_small_fold - 1 - i]
                : small_primes[i];
            uint32_t max_pat = (pat_size + margin) * 30;
            uint32_t pi = i < n_small_fold ? std::min(i, n_small_fold - 1 - i)
                                           : i - n_small_fold / 2;
            length[pi] = (uint8_t)pat_size;
            ilen[pi] = (uint8_t)(vec_size % pat_size);
            if (i && (i < n_small_fold / 2 || i >= n_small_fold))
                offset[pi] = offset[pi - 1] + length[pi - 1] + margin;
            uint8_t* pat = pattern + offset[pi];
            for (uint32_t j = small_primes[i]; j < max_pat; j += small_primes[i]) {
                uint32_t li = j / 30;
                uint8_t bi = sieve_idx[j % 30];
                if (bi != 0xFF)
                    pat[li] &= (uint8_t)~(1u << bi);
            }
        }
    }
};

inline const small_table& get_small_table() {
    static const small_table table;
    return table;
}

inline void sieve_small(uint8_t* sieve, uint8_t* offsets) {
    std::memset(sieve, 0xFF, mini_block);
    const small_table& table = get_small_table();
    for (uint32_t i = 0; i < n_patterns; ++i) {
        const uint8_t* pat = table.pattern + table.offset[i];
        uint16_t off = offsets[i];
        uint8_t len = table.length[i];
        uint8_t delta = table.ilen[i];
        for (uint16_t j = 0; j < mini_block; j += vec_size) {
            (vec::load(sieve + j) & vec::load(pat + off)).store(sieve + j);
            off += delta;
            if (off >= len) off -= len;
        }
        offsets[i] = (uint8_t)off;
    }
}

inline void sieve_intermediate(uint8_t* block,
                               const uint32_t* primes,
                               uint32_t* offsets,
                               uint32_t n,
                               const wheel_data& wheel) {
    uint8_t masks[8];
    for (uint32_t i = 0; i < 8; ++i)
        masks[i] = (uint8_t)~(1u << i);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t enc = primes[i];
        uint32_t N = enc / 8 * 8;
        uint32_t c = enc & 7;
        uint32_t j = offsets[i];
        for (; j < mini_block * 8; j += wheel.skip[c][j & 7] + N * wheel.skip_mul[c][j & 7])
            block[j / 8] &= masks[j & 7];
        offsets[i] = j - mini_block * 8;
    }
}

struct bucket {
    static constexpr uint32_t capacity = 1023;
    uint16_t length = 0;
    uint16_t data[capacity];

    void clear() { length = 0; }
    void push(uint16_t value) { data[length++] = value; }
    bool full() const { return length >= capacity; }
};

inline void bucket_sieve(uint8_t* sieve,
                         uint32_t window_bytes,
                         const uint32_t* primes,
                         uint32_t* offsets,
                         uint32_t p_start,
                         uint32_t p_end,
                         const wheel_data& wheel,
                         bucket* buckets,
                         uint32_t n_buckets) {
    if (p_start >= p_end || window_bytes == 0) return;

    uint32_t window_bits = window_bytes * 8;
    uint8_t masks[8];
    for (uint32_t i = 0; i < 8; ++i)
        masks[i] = (uint8_t)~(1u << i);

    uint32_t p_ind = p_start;
    while (p_ind < p_end) {
        for (uint32_t i = 0; i < n_buckets; ++i)
            buckets[i].clear();
        bool any_full = false;
        while (!any_full && p_ind < p_end) {
            uint32_t enc = primes[p_ind];
            uint32_t N = enc / 8 * 8;
            uint32_t c = enc & 7;
            uint32_t j = offsets[p_ind];
            bool finished_prime = true;
            while (j < window_bits) {
                bucket& bkt = buckets[j >> 16];
                bkt.push((uint16_t)(j & 0xFFFF));
                j += wheel.skip[c][j & 7] + N * wheel.skip_mul[c][j & 7];
                if (bkt.full()) {
                    any_full = true;
                    finished_prime = false;
                    break;
                }
            }
            if (finished_prime) {
                offsets[p_ind] = j - window_bits;
                ++p_ind;
            } else {
                offsets[p_ind] = j;
            }
        }

        for (uint32_t bi = 0; bi < n_buckets; ++bi) {
            uint8_t* block = sieve + bi * 8192;
            bucket& bkt = buckets[bi];
            for (uint16_t k = 0; k < bkt.length; ++k) {
                uint16_t off = bkt.data[k];
                block[off >> 3] &= masks[off & 7];
            }
        }
    }
}

inline void harvest_block(uint8_t* block, uint32_t bytes,
                          uint32_t base_enc, std::vector<uint32_t>& out) {
    for (uint32_t byte = 0; byte < bytes; ++byte) {
        uint8_t bits = block[byte];
        while (bits) {
            int bit = ctz_u64(bits);
            bits &= bits - 1;
            out.push_back(base_enc + byte * 8 + bit);
        }
    }
}

class base_prime_table {
public:
    const wheel_data& wheel() const { return wheel_; }
    const std::vector<uint32_t>& primes() const { return primes_; }

    uint32_t ensure_upto(uint32_t limit_bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        uint32_t enc_limit = limit_bytes * 8;
        if (primes_.empty() || primes_.back() < enc_limit) {
            uint64_t target = align_up(limit_bytes, (uint64_t)mini_block);
            extend_to(target);
        }
        auto it = std::lower_bound(primes_.begin(), primes_.end(), enc_limit);
        return (uint32_t)(it - primes_.begin());
    }

private:
    wheel_data wheel_;
    std::vector<uint32_t> primes_;
    uint64_t upper_bound_ = 0;
    std::mutex mutex_;

    base_prime_table() {
        std::vector<uint8_t> buffer(mini_block);
        uint8_t small_offsets[n_patterns] = {};

        sieve_small(buffer.data(), small_offsets);
        buffer[0] &= 0xFE;

        uint64_t block_end = (uint64_t)mini_block * 30;
        std::vector<uint32_t> dense_primes;
        std::vector<uint32_t> dense_offsets;
        for (uint32_t byte = 0; byte < mini_block; ++byte) {
            uint8_t bits = buffer[byte];
            while (bits) {
                int bit = ctz_u64(bits);
                bits &= bits - 1;
                uint32_t enc = byte * 8 + bit;
                uint64_t p = decode(enc);
                if (p * p > block_end)
                    goto bootstrap_done;
                dense_primes.push_back(enc);
                dense_offsets.push_back((uint32_t)wheel_.sqr(enc));
            }
        }
bootstrap_done:
        sieve_intermediate(buffer.data(), dense_primes.data(),
                           dense_offsets.data(), (uint32_t)dense_primes.size(),
                           wheel_);
        harvest_block(buffer.data(), mini_block, 0, primes_);
        upper_bound_ = mini_block;
    }

    void extend_to(uint64_t target_ub) {
        if (target_ub <= upper_bound_) return;

        std::vector<uint8_t> buffer(mini_block);
        uint8_t small_offsets[n_patterns];
        const small_table& table = get_small_table();
        for (uint32_t i = 0; i < n_patterns; ++i)
            small_offsets[i] = (uint8_t)(upper_bound_ % table.length[i]);

        std::vector<uint32_t> dense_primes;
        std::vector<uint32_t> dense_offsets;
        uint32_t scan = 0;

        auto grow_dense = [&](uint64_t block_end_num) {
            while (scan < primes_.size()) {
                uint32_t enc = primes_[scan];
                uint64_t p = decode(enc);
                if (p * p > block_end_num)
                    break;
                dense_primes.push_back(enc);
                dense_offsets.push_back((uint32_t)(
                    advance_to_at_least(wheel_.sqr(enc), upper_bound_ * 8, enc, wheel_) - upper_bound_ * 8));
                ++scan;
            }
        };

        while (upper_bound_ < target_ub) {
            uint64_t block_end = (upper_bound_ + mini_block) * 30;
            grow_dense(block_end);
            sieve_small(buffer.data(), small_offsets);
            sieve_intermediate(buffer.data(), dense_primes.data(),
                               dense_offsets.data(),
                               (uint32_t)dense_primes.size(), wheel_);
            harvest_block(buffer.data(), mini_block, (uint32_t)(upper_bound_ * 8), primes_);
            upper_bound_ += mini_block;
        }
    }

    friend base_prime_table& get_base_prime_table();
};

inline base_prime_table& get_base_prime_table() {
    static base_prime_table table;
    return table;
}

struct config {
    uint32_t num_threads = 0;
    uint32_t window_bytes = 0;
    uint32_t arch_window_bytes = default_window_bytes;
    uint32_t target_thread_bytes = default_target_thread_bytes;
};

struct query_plan {
    uint64_t lo;
    uint64_t hi;
    uint32_t threads;
    uint32_t window_bytes;

    query_plan(uint64_t lo_, uint64_t hi_, const config& cfg)
        : lo(lo_), hi(hi_), threads(1), window_bytes(mini_block) {
        uint64_t sieve_lo = std::max(lo, (uint64_t)229);
        uint64_t total_bytes = hi > sieve_lo ? div_ceil(hi - sieve_lo, (uint64_t)30) : 0;
        threads = choose_threads(total_bytes, cfg);
        window_bytes = choose_window(total_bytes, threads, cfg);
    }

    static uint32_t choose_threads(uint64_t total_bytes, const config& cfg) {
        uint32_t requested = cfg.num_threads ? cfg.num_threads : std::thread::hardware_concurrency();
        if (requested == 0) requested = 1;
        if (total_bytes == 0) return 1;
        uint64_t by_work = std::max<uint64_t>(1, total_bytes / std::max<uint32_t>(mini_block, cfg.target_thread_bytes));
        return (uint32_t)std::min<uint64_t>(requested, by_work);
    }

    static uint32_t choose_window(uint64_t total_bytes, uint32_t threads, const config& cfg) {
        uint64_t arch_default = std::max<uint32_t>(mini_block, cfg.arch_window_bytes);
        if (cfg.window_bytes != 0)
            return (uint32_t)align_up(std::max<uint32_t>(mini_block, cfg.window_bytes), (uint64_t)mini_block);
        if (total_bytes == 0)
            return mini_block;
        uint64_t per_thread = div_ceil(total_bytes, threads);
        uint64_t tuned = std::min<uint64_t>(arch_default, std::max<uint64_t>(mini_block, align_up(per_thread, (uint64_t)mini_block)));
        return (uint32_t)tuned;
    }
};

template<class UInt>
inline void collect_window_range(const uint8_t* sieve,
                                 uint32_t window_bytes,
                                 uint64_t window_byte_lo,
                                 uint64_t query_lo,
                                 uint64_t query_hi,
                                 std::vector<UInt>& out) {
    uint64_t num_lo = window_byte_lo * 30;
    uint64_t num_hi = num_lo + (uint64_t)window_bytes * 30;
    uint64_t eff_lo = std::max(num_lo, query_lo);
    uint64_t eff_hi = std::min(num_hi, query_hi);
    if (eff_lo >= eff_hi) return;

    uint32_t first_byte = (uint32_t)((eff_lo - num_lo) / 30);
    uint32_t last_byte = (uint32_t)((eff_hi - num_lo) / 30);
    if (last_byte > window_bytes) last_byte = window_bytes;

    auto append_masked = [&](uint32_t byte_index, uint8_t mask) {
        uint8_t bits = sieve[byte_index] & mask;
        uint64_t base = num_lo + (uint64_t)byte_index * 30;
        while (bits) {
            int bit = ctz_u64(bits);
            bits &= bits - 1;
            uint64_t prime = base + sieve_pos[bit];
            if (prime >= eff_lo && prime < eff_hi)
                out.push_back((UInt)prime);
        }
    };

    uint8_t first_mask = 0xFF;
    {
        uint64_t base = num_lo + (uint64_t)first_byte * 30;
        for (uint8_t bit = 0; bit < 8; ++bit)
            if (base + sieve_pos[bit] < eff_lo)
                first_mask &= (uint8_t)~(1u << bit);
    }

    uint8_t last_mask = 0xFF;
    if (last_byte < window_bytes) {
        last_mask = 0;
        uint64_t base = num_lo + (uint64_t)last_byte * 30;
        for (uint8_t bit = 0; bit < 8; ++bit)
            if (base + sieve_pos[bit] < eff_hi)
                last_mask |= (uint8_t)(1u << bit);
    }

    if (first_byte == last_byte) {
        append_masked(first_byte, first_mask & last_mask);
        return;
    }

    append_masked(first_byte, first_mask);
    for (uint32_t byte = first_byte + 1; byte < last_byte; ++byte) {
        uint8_t bits = sieve[byte];
        uint64_t base = num_lo + (uint64_t)byte * 30;
        while (bits) {
            int bit = ctz_u64(bits);
            bits &= bits - 1;
            out.push_back((UInt)(base + sieve_pos[bit]));
        }
    }
    if (last_byte < window_bytes)
        append_masked(last_byte, last_mask);
}

inline uint64_t count_window_range(const uint8_t* sieve,
                                   uint32_t window_bytes,
                                   uint64_t window_byte_lo,
                                   uint64_t query_lo,
                                   uint64_t query_hi) {
    uint64_t num_lo = window_byte_lo * 30;
    uint64_t num_hi = num_lo + (uint64_t)window_bytes * 30;
    uint64_t eff_lo = std::max(num_lo, query_lo);
    uint64_t eff_hi = std::min(num_hi, query_hi);
    if (eff_lo >= eff_hi) return 0;

    uint32_t first_byte = (uint32_t)((eff_lo - num_lo) / 30);
    uint32_t last_byte = (uint32_t)((eff_hi - num_lo) / 30);
    if (last_byte > window_bytes) last_byte = window_bytes;

    uint8_t first_mask = 0xFF;
    {
        uint64_t base = num_lo + (uint64_t)first_byte * 30;
        for (uint8_t bit = 0; bit < 8; ++bit)
            if (base + sieve_pos[bit] < eff_lo)
                first_mask &= (uint8_t)~(1u << bit);
    }

    uint8_t last_mask = 0;
    if (last_byte < window_bytes) {
        uint64_t base = num_lo + (uint64_t)last_byte * 30;
        for (uint8_t bit = 0; bit < 8; ++bit)
            if (base + sieve_pos[bit] < eff_hi)
                last_mask |= (uint8_t)(1u << bit);
    }

    if (first_byte == last_byte)
        return popcount_u64(sieve[first_byte] & first_mask & last_mask);
    if (first_byte > last_byte)
        return 0;

    uint64_t count = popcount_u64(sieve[first_byte] & first_mask);
    uint32_t mid_start = first_byte + 1;
    uint32_t mid_end = last_byte;
    if (mid_start < mid_end) {
        const uint64_t* words = (const uint64_t*)(sieve + mid_start);
        uint32_t n_bytes = mid_end - mid_start;
        count += popcount_array(words, n_bytes / 8);
        uint32_t tail = n_bytes & 7;
        if (tail) {
            uint64_t last = 0;
            std::memcpy(&last, sieve + mid_end - tail, tail);
            count += popcount_u64(last);
        }
    }

    if (last_mask && last_byte < window_bytes)
        count += popcount_u64(sieve[last_byte] & last_mask);
    return count;
}

struct count_consumer {
    uint64_t query_lo;
    uint64_t query_hi;
    uint64_t total = 0;

    void operator()(const uint8_t* sieve, uint32_t window_bytes, uint64_t window_byte_lo) {
        total += count_window_range(sieve, window_bytes, window_byte_lo, query_lo, query_hi);
    }
};

template<class UInt>
struct collect_consumer {
    uint64_t query_lo;
    uint64_t query_hi;
    std::vector<UInt> values;

    void operator()(const uint8_t* sieve, uint32_t window_bytes, uint64_t window_byte_lo) {
        collect_window_range(sieve, window_bytes, window_byte_lo, query_lo, query_hi, values);
    }
};

struct worker_chunk {
    uint64_t query_lo;
    uint64_t query_hi;
};

inline std::vector<worker_chunk> split_query(uint64_t lo, uint64_t hi, uint32_t threads) {
    std::vector<worker_chunk> chunks;
    uint64_t start = std::max(lo, (uint64_t)229);
    if (hi <= start || threads == 0) return chunks;
    chunks.reserve(threads);
    uint64_t span = hi - start;
    for (uint32_t i = 0; i < threads; ++i) {
        uint64_t sub_lo = start + span * i / threads;
        uint64_t sub_hi = start + span * (i + 1) / threads;
        if (sub_lo < sub_hi)
            chunks.push_back({sub_lo, sub_hi});
    }
    return chunks;
}

struct worker_ctx {
    const base_prime_table* base = nullptr;
    const query_plan* plan = nullptr;
    worker_chunk chunk{};

    uint64_t sieve_byte_lo = 0;
    uint64_t sieve_byte_hi = 0;
    uint32_t window_bytes = mini_block;
    uint32_t required_np = 0;
    uint32_t dense_end = 0;
    uint32_t super_start = 0;

    std::vector<uint8_t> sieve;
    std::vector<uint32_t> offsets;
    std::vector<bucket> buckets;
    std::array<uint8_t, n_patterns> small_offsets{};

    std::vector<int32_t> super_heads;
    std::vector<int32_t> super_links;
    std::vector<uint64_t> super_hits;

    worker_ctx(const base_prime_table& base_table,
               const query_plan& query,
               worker_chunk worker_chunk_)
        : base(&base_table), plan(&query), chunk(worker_chunk_) {
        if (chunk.query_hi <= chunk.query_lo)
            return;

        sieve_byte_lo = (chunk.query_lo / 30) / mini_block * mini_block;
        sieve_byte_hi = align_up(div_ceil(chunk.query_hi, (uint64_t)30), (uint64_t)mini_block);
        window_bytes = plan->window_bytes;
        sieve.resize(window_bytes);
        buckets.resize(window_bytes / 8192);

        const small_table& table = get_small_table();
        for (uint32_t i = 0; i < n_patterns; ++i)
            small_offsets[i] = (uint8_t)(sieve_byte_lo % table.length[i]);

        init_bands();
        init_offsets();
        seed_super();
    }

    template<class Consumer>
    void run(Consumer& consumer) {
        if (chunk.query_hi <= chunk.query_lo)
            return;

        for (uint64_t window_lo = sieve_byte_lo, wi = 0;
             window_lo < sieve_byte_hi;
             window_lo += window_bytes, ++wi) {
            uint32_t bytes = (uint32_t)std::min<uint64_t>(window_bytes, sieve_byte_hi - window_lo);

            for (uint32_t b = 0; b < bytes; b += mini_block) {
                uint8_t* block = sieve.data() + b;
                sieve_small(block, small_offsets.data());
                sieve_intermediate(block,
                                   base->primes().data(),
                                   offsets.data(),
                                   dense_end,
                                   base->wheel());
            }

            if (super_start > dense_end) {
                bucket_sieve(sieve.data(), bytes,
                             base->primes().data(), offsets.data(),
                             dense_end, super_start,
                             base->wheel(),
                             buckets.data(), bytes / 8192);
            }

            apply_super(window_lo * 8, wi, bytes);
            if (window_lo == 0)
                sieve[0] &= 0xFE;

            consumer(sieve.data(), bytes, window_lo);

            if (bytes < window_bytes)
                break;
        }
    }

private:
    void init_bands() {
        uint64_t sqrt_hi = chunk.query_hi > 0 ? isqrt(chunk.query_hi - 1) : 0;
        uint32_t enc_limit = (uint32_t)((sqrt_hi / 30 + 1) * 8);
        const std::vector<uint32_t>& primes = base->primes();
        required_np = (uint32_t)(std::lower_bound(primes.begin(), primes.end(), enc_limit) - primes.begin());

        uint32_t dense_limit_prime = std::min<uint32_t>((uint32_t)(window_bytes * 30), default_dense_prime);
        uint32_t dense_limit_enc = (dense_limit_prime / 30) * 8;
        uint32_t super_limit_enc = window_bytes * 8;

        dense_end = (uint32_t)(std::lower_bound(primes.begin(), primes.begin() + required_np, dense_limit_enc) - primes.begin());
        super_start = (uint32_t)(std::lower_bound(primes.begin() + dense_end, primes.begin() + required_np, super_limit_enc) - primes.begin());
        if (super_start < dense_end)
            super_start = dense_end;
    }

    void init_offsets() {
        offsets.resize(super_start);
        uint64_t base_enc = sieve_byte_lo * 8;
        const std::vector<uint32_t>& primes = base->primes();
        for (uint32_t i = 0; i < super_start; ++i) {
            uint32_t enc = primes[i];
            uint64_t hit = advance_to_at_least(base->wheel().sqr(enc), base_enc, enc, base->wheel());
            offsets[i] = (uint32_t)(hit - base_enc);
        }
    }

    void seed_super() {
        uint32_t n_super = required_np - super_start;
        if (n_super == 0) return;

        uint64_t chunk_enc_lo = sieve_byte_lo * 8;
        uint64_t chunk_enc_hi = sieve_byte_hi * 8;
        uint64_t window_bits = (uint64_t)window_bytes * 8;
        uint64_t window_count = div_ceil(sieve_byte_hi - sieve_byte_lo, (uint64_t)window_bytes);

        super_heads.assign((size_t)window_count, -1);
        super_links.resize(n_super);
        super_hits.resize(n_super);

        const std::vector<uint32_t>& primes = base->primes();
        for (uint32_t local = 0; local < n_super; ++local) {
            uint32_t enc = primes[super_start + local];
            uint64_t hit = advance_to_at_least(base->wheel().sqr(enc), chunk_enc_lo, enc, base->wheel());
            if (hit >= chunk_enc_hi) {
                super_links[local] = -1;
                continue;
            }
            super_hits[local] = hit;
            uint64_t wi = (hit - chunk_enc_lo) / window_bits;
            super_links[local] = super_heads[(size_t)wi];
            super_heads[(size_t)wi] = (int32_t)local;
        }
    }

    void apply_super(uint64_t window_enc_lo, uint64_t window_index, uint32_t bytes) {
        if (super_heads.empty()) return;

        uint64_t chunk_enc_lo = sieve_byte_lo * 8;
        uint64_t chunk_enc_hi = sieve_byte_hi * 8;
        uint64_t window_bits = (uint64_t)bytes * 8;
        uint8_t masks[8];
        for (uint32_t i = 0; i < 8; ++i)
            masks[i] = (uint8_t)~(1u << i);

        int32_t idx = super_heads[(size_t)window_index];
        super_heads[(size_t)window_index] = -1;
        while (idx != -1) {
            int32_t next = super_links[(size_t)idx];
            uint64_t hit = super_hits[(size_t)idx];
            uint64_t rel = hit - window_enc_lo;
            if (rel < window_bits)
                sieve[(size_t)(rel >> 3)] &= masks[rel & 7];

            uint32_t enc = base->primes()[super_start + (uint32_t)idx];
            hit = advance_hit(hit, enc, base->wheel());
            if (hit < chunk_enc_hi) {
                super_hits[(size_t)idx] = hit;
                uint64_t wi = (hit - chunk_enc_lo) / ((uint64_t)window_bytes * 8);
                super_links[(size_t)idx] = super_heads[(size_t)wi];
                super_heads[(size_t)wi] = idx;
            }

            idx = next;
        }
    }
};

inline void prewarm_base_primes(uint64_t hi) {
    if (hi <= 229) return;
    get_base_prime_table().ensure_upto((uint32_t)(isqrt(hi - 1) / 30 + 1));
}

template<class UInt>
inline void sieve(uint64_t lo, uint64_t hi, std::vector<UInt>& out, config cfg = {}) {
    out.clear();
    if (hi <= lo) return;

    for (uint32_t i = 0; i < n_presieve; ++i) {
        uint64_t p = presieve_primes[i];
        if (p >= hi) break;
        if (p >= lo) out.push_back((UInt)p);
    }

    uint64_t sieve_lo = std::max(lo, (uint64_t)229);
    if (hi <= sieve_lo) return;

    query_plan plan(lo, hi, cfg);
    prewarm_base_primes(hi);

    std::vector<worker_chunk> chunks = split_query(lo, hi, plan.threads);
    std::vector<std::vector<UInt>> partials(chunks.size());
    std::vector<std::thread> workers;
    std::exception_ptr error;
    std::mutex error_lock;
    workers.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
        workers.emplace_back([&, i]() {
            try {
                worker_ctx ctx(get_base_prime_table(), plan, chunks[i]);
                collect_consumer<UInt> consumer{chunks[i].query_lo, chunks[i].query_hi, {}};
                ctx.run(consumer);
                partials[i] = std::move(consumer.values);
            } catch (...) {
                std::lock_guard<std::mutex> guard(error_lock);
                if (!error) error = std::current_exception();
            }
        });
    }

    for (auto& worker : workers)
        worker.join();
    if (error) std::rethrow_exception(error);

    size_t total = out.size();
    for (const auto& part : partials) total += part.size();
    out.reserve(total);
    for (auto& part : partials)
        out.insert(out.end(), part.begin(), part.end());
}

inline void sieve(uint32_t lo, uint32_t hi, std::vector<uint32_t>& out, config cfg = {}) {
    sieve<uint32_t>((uint64_t)lo, (uint64_t)hi, out, cfg);
}

inline uint64_t count_primes(uint64_t lo, uint64_t hi, config cfg = {}) {
    if (hi <= lo) return 0;
    uint64_t total = count_presieve(lo, hi);
    uint64_t sieve_lo = std::max(lo, (uint64_t)229);
    if (hi <= sieve_lo) return total;

    query_plan plan(lo, hi, cfg);
    prewarm_base_primes(hi);

    std::vector<worker_chunk> chunks = split_query(lo, hi, plan.threads);
    std::vector<uint64_t> partials(chunks.size(), 0);
    std::vector<std::thread> workers;
    std::exception_ptr error;
    std::mutex error_lock;
    workers.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
        workers.emplace_back([&, i]() {
            try {
                worker_ctx ctx(get_base_prime_table(), plan, chunks[i]);
                count_consumer consumer{chunks[i].query_lo, chunks[i].query_hi, 0};
                ctx.run(consumer);
                partials[i] = consumer.total;
            } catch (...) {
                std::lock_guard<std::mutex> guard(error_lock);
                if (!error) error = std::current_exception();
            }
        });
    }

    for (auto& worker : workers)
        worker.join();
    if (error) std::rethrow_exception(error);

    for (uint64_t part : partials)
        total += part;
    return total;
}

inline uint64_t count_primes(uint64_t hi, config cfg = {}) {
    return count_primes(2, hi, cfg);
}

struct prime_sieve {
    uint64_t lo;
    uint64_t hi;
    config cfg;
    std::vector<uint64_t> cached;
    size_t index = 0;
    bool loaded = false;

    prime_sieve(uint64_t lo_, uint64_t hi_, config cfg_ = {})
        : lo(lo_), hi(hi_), cfg(cfg_) {}

    uint64_t next_prime() {
        if (!loaded) {
            sieve<uint64_t>(lo, hi, cached, cfg);
            loaded = true;
        }
        if (index >= cached.size()) return 0;
        return cached[index++];
    }

    void collect(std::vector<uint64_t>& out) {
        if (!loaded) {
            sieve<uint64_t>(lo, hi, cached, cfg);
            loaded = true;
        }
        out.insert(out.end(), cached.begin() + (ptrdiff_t)index, cached.end());
        index = cached.size();
    }

    uint64_t count_primes() const {
        return sieve_v3::count_primes(lo, hi, cfg);
    }

    static uint64_t pi(uint64_t n, config cfg = {}) {
        if (n <= 2) return 0;
        return sieve_v3::count_primes(2, n, cfg);
    }

    static std::vector<uint64_t> range(uint64_t lo, uint64_t hi, config cfg = {}) {
        std::vector<uint64_t> out;
        sieve<uint64_t>(lo, hi, out, cfg);
        return out;
    }
};

using prime_sieve_v3 = prime_sieve;

} // namespace sieve_v3
} // namespace zfactor
