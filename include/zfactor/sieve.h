#pragma once

// zfactor::sieve — thin wrapper over Kim Walisch's primesieve library.
//
// Exposes the surface zfactor needs (range, count, nth_prime, iteration) in a
// libfact-shaped namespace so callers don't depend on primesieve's headers
// directly.  All numbers are uint64_t; full 64-bit range is supported.

#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>

#include <primesieve.hpp>
#include <primesieve/iterator.hpp>

namespace zfactor {

// Forward iterator over primes in [start, stop).  Cheap to construct, cheap
// to advance — primesieve buffers ~1024 primes per refill.
class PrimeIter {
public:
    PrimeIter() noexcept : it_(), stop_(0) {}

    // Iterate over primes p with start <= p < stop.
    PrimeIter(uint64_t start, uint64_t stop) noexcept
        : it_(start, stop == 0 ? 0 : stop - 1), stop_(stop) {}

    // Returns the next prime in [start, stop), or 0 when exhausted.
    uint64_t next() noexcept {
        uint64_t p = it_.next_prime();
        return p < stop_ ? p : 0;
    }

    // Resets to a new range without freeing internal buffers.
    void reset(uint64_t start, uint64_t stop) noexcept {
        stop_ = stop;
        it_.jump_to(start, stop == 0 ? 0 : stop - 1);
    }

private:
    primesieve::iterator it_;
    uint64_t stop_;
};

// All primes p with start <= p < stop, in ascending order.  Output element
// type is templated — primesieve::generate_primes natively fills u16, u32,
// or u64 vectors, so callers that know primes fit in 32 bits can ask for a
// `std::vector<uint32_t>` directly and skip the widen-on-load.
template<typename T = uint64_t>
inline std::vector<T> primes_range_as(uint64_t start, uint64_t stop) {
    std::vector<T> v;
    if (stop <= start || stop < 2) return v;
    primesieve::generate_primes(start, stop - 1, &v);
    return v;
}

inline std::vector<uint64_t> primes_range(uint64_t start, uint64_t stop) {
    return primes_range_as<uint64_t>(start, stop);
}

// All primes p with p < limit, in ascending order.
inline std::vector<uint64_t> primes_below(uint64_t limit) {
    return primes_range(0, limit);
}

// pi(stop) - pi(start) — number of primes p with start <= p < stop.
inline uint64_t prime_count(uint64_t start, uint64_t stop) {
    if (stop <= start || stop < 2) return 0;
    return primesieve::count_primes(start, stop - 1);
}

// pi(limit) — number of primes p < limit.
inline uint64_t prime_count(uint64_t limit) {
    return prime_count(0, limit);
}

// The n-th prime, 1-indexed: nth_prime(1)==2, nth_prime(2)==3, ...
inline uint64_t nth_prime(uint64_t n) {
    return primesieve::nth_prime(static_cast<int64_t>(n));
}

// Stream all primes p with start <= p < stop to `fn`.
// fn signature: void(uint64_t prime).
template<typename F>
inline void for_each_prime(uint64_t start, uint64_t stop, F&& fn) {
    if (stop <= start || stop < 2) return;
    PrimeIter it(start, stop);
    for (uint64_t p; (p = it.next()) != 0; ) fn(p);
}

template<typename F>
inline void for_each_prime(uint64_t limit, F&& fn) {
    for_each_prime(uint64_t(0), limit, std::forward<F>(fn));
}

} // namespace zfactor
