#pragma once

#include "montgomery.h"

namespace zfactor::fixint {

// --- Thread-local context stack ---

template<int N>
struct MontCtxStack {
    static constexpr int MAX_DEPTH = 4;
    const MontCtx<N>* stack[MAX_DEPTH] = {};
    int depth = 0;

    void push(const MontCtx<N>* c) { stack[depth++] = c; }
    void pop() { --depth; }
    const MontCtx<N>& top() const { return *stack[depth - 1]; }
};

template<int N>
inline thread_local MontCtxStack<N> _ctx_stack;

template<int N>
inline const MontCtx<N>& ctx() { return _ctx_stack<N>.top(); }

template<int N>
struct [[nodiscard]] MontScope {
    MontScope(const MontCtx<N>& c) { _ctx_stack<N>.push(&c); }
    ~MontScope() { _ctx_stack<N>.pop(); }
    MontScope(const MontScope&) = delete;
    MontScope& operator=(const MontScope&) = delete;
};

// --- Forward declarations ---

template<int N> struct Mod;
template<int N> struct ModWide;

// --- Mod<N>: fully reduced Montgomery form, value in [0, mod) ---

template<int N>
struct Mod {
    UInt<N> v;

    // Construct Montgomery 1 from context
    static Mod one() {
        Mod r;
        r.v = ctx<N>().r_mod;
        return r;
    }

    // Convert plain integer to Montgomery form
    static Mod from_uint(const UInt<N>& a) {
        Mod r;
        to_mont<N>(r.v.d, a.d, ctx<N>());
        return r;
    }

    // Convert back from Montgomery form to plain integer
    UInt<N> to_uint() const {
        UInt<N> r;
        from_mont<N>(r.d, v.d, ctx<N>());
        return r;
    }

    // Montgomery multiplication (CIOS)
    friend Mod operator*(const Mod& a, const Mod& b) {
        Mod r;
        montmul<N>(r.v.d, a.v.d, b.v.d, ctx<N>());
        return r;
    }

    Mod& operator*=(const Mod& other) {
        montmul<N>(v.d, v.d, other.v.d, ctx<N>());
        return *this;
    }

    // Modular squaring (optimized sqr + REDC)
    Mod sqr() const {
        Mod r;
        montsqr<N>(r.v.d, v.d, ctx<N>());
        return r;
    }

    // Modular addition: add + reduce
    // When add overflows N limbs (cy=1), true sum >= 2^(64N) > mod, subtract unconditionally.
    // When no overflow, conditionally subtract.
    friend Mod operator+(const Mod& a, const Mod& b) {
        Mod r;
        uint8_t cy = mpn::add<N>(r.v.d, a.v.d, b.v.d);
        if (cy)
            mpn::sub<N>(r.v.d, r.v.d, ctx<N>().mod.d);
        else
            mpn::csub<N>(r.v.d, r.v.d, ctx<N>().mod.d);
        return r;
    }

    Mod& operator+=(const Mod& other) {
        uint8_t cy = mpn::add<N>(v.d, v.d, other.v.d);
        if (cy)
            mpn::sub<N>(v.d, v.d, ctx<N>().mod.d);
        else
            mpn::csub<N>(v.d, v.d, ctx<N>().mod.d);
        return *this;
    }

    // Modular subtraction: sub + cadd if borrow
    friend Mod operator-(const Mod& a, const Mod& b) {
        Mod r;
        mpn::limb_t bw = mpn::sub<N>(r.v.d, a.v.d, b.v.d);
        mpn::cadd<N>(r.v.d, r.v.d, ctx<N>().mod.d, bw);
        return r;
    }

    Mod& operator-=(const Mod& other) {
        mpn::limb_t bw = mpn::sub<N>(v.d, v.d, other.v.d);
        mpn::cadd<N>(v.d, v.d, ctx<N>().mod.d, bw);
        return *this;
    }

    bool operator==(const Mod& other) const {
        return mpn::cmp<N>(v.d, other.v.d) == 0;
    }

    bool operator!=(const Mod& other) const {
        return !(*this == other);
    }
};

// --- ModWide<N>: unreduced wide value for accumulation patterns ---

template<int N>
struct ModWide {
    UIntWide<N> v;

    // Montgomery reduction
    Mod<N> redc() const {
        Mod<N> r;
        mont_redc<N>(r.v.d, v.data(), ctx<N>());
        return r;
    }

    friend ModWide operator+(const ModWide& a, const ModWide& b) {
        ModWide r;
        mpn::add<2 * N>(r.v.data(), a.v.data(), b.v.data());
        return r;
    }

    friend ModWide operator-(const ModWide& a, const ModWide& b) {
        ModWide r;
        mpn::sub<2 * N>(r.v.data(), a.v.data(), b.v.data());
        return r;
    }
};

// Explicit wide multiply (no REDC) for accumulation patterns
template<int N>
inline ModWide<N> mul_wide(const Mod<N>& a, const Mod<N>& b) {
    ModWide<N> r;
    mpn::mul<N>(r.v.data(), a.v.d, b.v.d);
    return r;
}

// --- Fused operations ---

// fmadd: (a * b) + c
template<int N>
inline Mod<N> fmadd(const Mod<N>& a, const Mod<N>& b, const Mod<N>& c) {
    return a * b + c;
}

// fmsub: (a * b) - c
template<int N>
inline Mod<N> fmsub(const Mod<N>& a, const Mod<N>& b, const Mod<N>& c) {
    return a * b - c;
}

// --- Modular exponentiation ---

// Binary square-and-multiply (fast for small N where montmul is cheap)
template<int N>
inline Mod<N> pow_binary(const Mod<N>& base, const UInt<N>& exp) {
    Mod<N> result = Mod<N>::one();
    Mod<N> b = base;
    unsigned bits = mpn::bit_length<N>(exp.d);
    for (unsigned i = 0; i < bits; ++i) {
        if (mpn::test_bit<N>(exp.d, i))
            result *= b;
        b = b.sqr();
    }
    return result;
}

// Fixed-window exponentiation (better for larger N where montmul is expensive)
template<int N>
inline Mod<N> pow_window(const Mod<N>& base, const UInt<N>& exp, int w) {
    unsigned n = mpn::bit_length<N>(exp.d);
    if (n == 0) return Mod<N>::one();
    if (n == 1) return base;

    int half = 1 << (w - 1);
    // Precompute odd powers: table[0]=g, table[1]=g^3, ..., table[half-1]=g^(2*half-1)
    Mod<N> table[16]; // max w=5 → half=16
    table[0] = base;
    Mod<N> sq = base.sqr();
    for (int i = 1; i < half; ++i)
        table[i] = table[i - 1] * sq;

    // Scan exponent MSB→LSB, collect windows at '1' bits
    Mod<N> acc{};
    bool started = false;
    int pos = static_cast<int>(n) - 1;

    while (pos >= 0) {
        if (!mpn::test_bit<N>(exp.d, static_cast<unsigned>(pos))) {
            if (started) acc = acc.sqr();
            --pos;
            continue;
        }
        // Found a '1' bit. Grab a w-bit window.
        int win_end = pos;
        int win_start = pos - w + 1;
        if (win_start < 0) win_start = 0;

        unsigned val = 0;
        for (int j = win_end; j >= win_start; --j)
            val = (val << 1) | (mpn::test_bit<N>(exp.d, static_cast<unsigned>(j)) ? 1 : 0);

        // Strip trailing zeros
        while (val > 0 && (val & 1) == 0) { val >>= 1; win_start++; }

        int win_bits = win_end - win_start + 1;

        if (started) {
            for (int j = 0; j < win_bits; ++j)
                acc = acc.sqr();
            acc *= table[(val - 1) / 2];
        } else {
            acc = table[(val - 1) / 2];
            started = true;
        }
        pos = win_start - 1;
    }

    return started ? acc : Mod<N>::one();
}

// Auto-select: binary for N<=2 (cheap montmul), window for N>=3
template<int N>
inline Mod<N> pow(const Mod<N>& base, const UInt<N>& exp) {
    if constexpr (N <= 2) {
        return pow_binary<N>(base, exp);
    } else {
        unsigned bits = mpn::bit_length<N>(exp.d);
        int w = (bits <= 64) ? 3 : (bits <= 256) ? 4 : 5;
        return pow_window<N>(base, exp, w);
    }
}

} // namespace zfactor::fixint
