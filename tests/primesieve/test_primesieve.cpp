#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "zfactor/sieve.h"

using zfactor::PrimeIter;
using zfactor::primes_range;
using zfactor::primes_below;
using zfactor::prime_count;
using zfactor::nth_prime;

// --- pi(n) counting ---

TEST_CASE("pi(10)") { CHECK(prime_count(10) == 4); }
TEST_CASE("pi(100)") { CHECK(prime_count(100) == 25); }
TEST_CASE("pi(1000)") { CHECK(prime_count(1000) == 168); }
TEST_CASE("pi(10000)") { CHECK(prime_count(10000) == 1229); }
TEST_CASE("pi(100000)") { CHECK(prime_count(100000) == 9592); }
TEST_CASE("pi(1000000)") { CHECK(prime_count(1000000) == 78498); }
TEST_CASE("pi(10000000)") { CHECK(prime_count(10000000) == 664579); }
TEST_CASE("pi(100000000)") { CHECK(prime_count(100000000) == 5761455); }

// --- edge cases ---

TEST_CASE("pi(0) and pi(1) and pi(2)") {
    CHECK(prime_count(0) == 0);
    CHECK(prime_count(1) == 0);
    CHECK(prime_count(2) == 0);
}

TEST_CASE("pi(3) = 1") { CHECK(prime_count(3) == 1); }
TEST_CASE("pi(6) = 3") { CHECK(prime_count(6) == 3); }

// --- range: small primes ---

TEST_CASE("primes in [0, 100)") {
    auto primes = primes_range(0, 100);
    std::vector<uint64_t> expected = {
        2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,
        53,59,61,67,71,73,79,83,89,97
    };
    CHECK(primes == expected);
}

TEST_CASE("primes in [220, 240)") {
    auto primes = primes_range(220, 240);
    std::vector<uint64_t> expected = {223, 227, 229, 233, 239};
    CHECK(primes == expected);
}

TEST_CASE("primes in [225, 235)") {
    auto primes = primes_range(225, 235);
    std::vector<uint64_t> expected = {227, 229, 233};
    CHECK(primes == expected);
}

// --- nth_prime ---

TEST_CASE("nth_prime") {
    CHECK(nth_prime(1) == 2);
    CHECK(nth_prime(2) == 3);
    CHECK(nth_prime(3) == 5);
    CHECK(nth_prime(4) == 7);
    CHECK(nth_prime(10) == 29);
    CHECK(nth_prime(100) == 541);
    CHECK(nth_prime(1000) == 7919);
    CHECK(nth_prime(10000) == 104729);
}

// --- iteration vs count consistency ---

TEST_CASE("iteration count matches pi for 10^6") {
    PrimeIter it(2, 1000000);
    uint64_t count = 0;
    while (it.next() != 0) ++count;
    CHECK(count == 78498);
}

// --- windowed count ---

TEST_CASE("count_primes([10^7, 10^8)) = pi(10^8) - pi(10^7)") {
    CHECK(prime_count(10000000, 100000000) == 5761455 - 664579);
}

TEST_CASE("count_primes([1000, 2000))") {
    auto primes = primes_range(1000, 2000);
    CHECK(prime_count(1000, 2000) == primes.size());
}

// --- large intervals (reference values from primesieve CLI) ---

TEST_CASE("large: [10^10, 10^10 + 10^8]") {
    constexpr uint64_t lo = 10000000000ULL;
    constexpr uint64_t hi = lo + 100000001ULL;
    CHECK(prime_count(lo, hi) == 4341930);
}

TEST_CASE("large: [10^11, 10^11 + 10^9]") {
    constexpr uint64_t lo = 100000000000ULL;
    constexpr uint64_t hi = lo + 1000000001ULL;
    CHECK(prime_count(lo, hi) == 39475591);
}

TEST_CASE("large: [10^12, 10^12 + 10^6]") {
    constexpr uint64_t lo = 1000000000000ULL;
    constexpr uint64_t hi = lo + 1000001ULL;
    CHECK(prime_count(lo, hi) == 36249);
}

TEST_CASE("large: [10^12, 10^12 + 10^8]") {
    constexpr uint64_t lo = 1000000000000ULL;
    constexpr uint64_t hi = lo + 100000001ULL;
    CHECK(prime_count(lo, hi) == 3618282);
}

// --- iteration vs count consistency on a large interval ---

TEST_CASE("large: iteration matches count for [10^12, 10^12 + 10^6]") {
    constexpr uint64_t lo = 1000000000000ULL;
    constexpr uint64_t hi = lo + 1000001ULL;

    PrimeIter it(lo, hi);
    uint64_t iter_count = 0;
    while (it.next() != 0) ++iter_count;

    CHECK(iter_count == prime_count(lo, hi));
    CHECK(iter_count == 36249);
}
