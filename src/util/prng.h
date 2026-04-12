#pragma once

// xoshiro256** — fast 256-bit-state PRNG by David Blackman and Sebastiano Vigna
// (2018), public domain.  Reference: https://prng.di.unimi.it/xoshiro256starstar.c
//
// Used by zfactor for Pollard rho cycle starts, Miller-Rabin witness selection,
// ECM curve parameters, and any other non-cryptographic randomness.
//
// Period 2^256 - 1.  Passes BigCrush.  ~1 ns / output on modern x86_64.
// NOT cryptographically secure — do not use for key material.

#include <cstdint>

namespace zfactor {

class Xoshiro256ss {
public:
    // Default-constructed generator is in a fixed but arbitrary good state.
    Xoshiro256ss() noexcept { seed(UINT64_C(0x9E3779B97F4A7C15)); }

    // Seed from a single 64-bit value.  Expands to 256 bits via splitmix64,
    // which is the seeding procedure recommended by the xoshiro authors.
    explicit Xoshiro256ss(uint64_t s) noexcept { seed(s); }

    void seed(uint64_t s) noexcept {
        // splitmix64 — expand 64 bits to four independent 64-bit words.
        auto sm64 = [&s]() noexcept -> uint64_t {
            s += UINT64_C(0x9E3779B97F4A7C15);
            uint64_t z = s;
            z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
            return z ^ (z >> 31);
        };
        state_[0] = sm64();
        state_[1] = sm64();
        state_[2] = sm64();
        state_[3] = sm64();
    }

    // One 64-bit output.  O(1), branchless.
    uint64_t next() noexcept {
        const uint64_t result = rotl(state_[1] * 5, 7) * 9;
        const uint64_t t = state_[1] << 17;

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];

        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);

        return result;
    }

    // Advance the state by 2^128 calls to next().  Useful for partitioning the
    // sequence across independent streams / threads without overlap.
    void jump() noexcept {
        static constexpr uint64_t JUMP[4] = {
            UINT64_C(0x180ec6d33cfd0aba),
            UINT64_C(0xd5a61266f0c9392c),
            UINT64_C(0xa9582618e03fc9aa),
            UINT64_C(0x39abdc4529b1661c),
        };
        apply_poly(JUMP);
    }

    // Advance the state by 2^192 calls to next().
    void long_jump() noexcept {
        static constexpr uint64_t LONG_JUMP[4] = {
            UINT64_C(0x76e15d3efefdcbbf),
            UINT64_C(0xc5004e441c522fb3),
            UINT64_C(0x77710069854ee241),
            UINT64_C(0x39109bb02acbe635),
        };
        apply_poly(LONG_JUMP);
    }

private:
    static constexpr uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    void apply_poly(const uint64_t (&poly)[4]) noexcept {
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; ++i) {
            for (int b = 0; b < 64; ++b) {
                if (poly[i] & (UINT64_C(1) << b)) {
                    s0 ^= state_[0];
                    s1 ^= state_[1];
                    s2 ^= state_[2];
                    s3 ^= state_[3];
                }
                (void)next();
            }
        }
        state_[0] = s0;
        state_[1] = s1;
        state_[2] = s2;
        state_[3] = s3;
    }

    uint64_t state_[4];
};

} // namespace zfactor
