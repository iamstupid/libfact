#pragma once

#include <cstdint>
#include <vector>
#include "uint.h"

namespace zfactor::fixint {

// ============================================================================
// Windowed exponentiation plan.
//
// Table: odd powers only. table[i] = g^(2i+1).
//   table[0]=g, table[1]=g^3, table[2]=g^5, ...
//
// seq layout: [digit, dbl_count, digit, dbl_count, digit, ...]
//   - Starts with a digit (initial load from table).
//   - Then alternating (dbl_count, digit) pairs.
//   - digit is 1-indexed into odd-power table:
//       digit d → table[d-1] = g^(2d-1).  d=1→g, d=2→g^3, d=3→g^5, ...
//   - dbl_count includes inter-window gap + trailing zeros from even window values.
//   - Trailing sentinel: dbl_count with digit=0 means just square, no multiply.
//
// Precompute cost: table_size-1 multiplies (g^2 computed once, then chain).
// ============================================================================

struct WindowedPlan {
    std::vector<uint16_t> seq;
    int table_size = 0;  // number of odd-power table entries actually needed

    template<int N>
    void plan(const UInt<N>& exp, int w) {
        seq.clear();
        table_size = 0;

        unsigned n = mpn::bit_length<N>(exp.d);
        if (n == 0) return;

        // Scan MSB→LSB: greedily grab w-bit windows starting at each '1' bit.
        // Each window: extract w bits, strip trailing zeros → odd value + extra dbls.
        struct Win { int pos; unsigned odd_val; int width; };
        std::vector<Win> wins;

        int pos = static_cast<int>(n) - 1;
        while (pos >= 0) {
            if (!mpn::test_bit<N>(exp.d, static_cast<unsigned>(pos))) { --pos; continue; }
            int end = pos;
            int start = pos - w + 1;
            if (start < 0) start = 0;
            unsigned val = 0;
            for (int j = end; j >= start; --j)
                val = (val << 1) | (mpn::test_bit<N>(exp.d, static_cast<unsigned>(j)) ? 1 : 0);
            while (val > 0 && (val & 1) == 0) { val >>= 1; start++; }
            wins.push_back({end, val, end - start + 1});
            pos = start - 1;
        }

        if (wins.empty()) return;

        auto digit = [](unsigned odd_val) -> uint8_t {
            return static_cast<uint16_t>((odd_val + 1) / 2);
        };

        // First window: just a digit
        seq.push_back(digit(wins[0].odd_val));

        // Subsequent: dbl_count + digit
        for (size_t i = 1; i < wins.size(); ++i) {
            int prev_bottom = wins[i-1].pos - wins[i-1].width;
            int dbls = prev_bottom - wins[i].pos + wins[i].width;
            seq.push_back(static_cast<uint16_t>(dbls));
            seq.push_back(digit(wins[i].odd_val));
        }

        // Trailing doublings to bit 0
        int remaining = wins.back().pos - wins.back().width + 1;
        if (remaining > 0) {
            seq.push_back(static_cast<uint16_t>(remaining));
            seq.push_back(0);
        }

        // Table size = max digit used (digits at positions 0, 2, 4, ...)
        if (!seq.empty()) table_size = seq[0];
        for (size_t i = 2; i < seq.size(); i += 2)
            if (seq[i] > table_size) table_size = seq[i];
    }

    template<typename Group>
    typename Group::T exec(const typename Group::T& base) const {
        using T = typename Group::T;
        // Precompute odd powers: table[0]=g, table[1]=g^3, table[2]=g^5, ...
        T table[32]; // w<=6 → max 32 entries
        table[0] = base;
        if (table_size > 1) {
            T sq = Group::dbl(base);
            for (int i = 1; i < table_size; ++i)
                table[i] = Group::add(table[i - 1], sq);
        }

        if (seq.empty()) return base;

        T acc = table[seq[0] - 1];

        for (size_t i = 1; i + 1 < seq.size(); i += 2) {
            int dbls = seq[i];
            for (int j = 0; j < dbls; ++j)
                acc = Group::dbl(acc);
            if (seq[i + 1] > 0)
                acc = Group::add(acc, table[seq[i + 1] - 1]);
        }

        return acc;
    }
};

// ============================================================================
// wNAF plan. Same interleaved layout but int8_t digits.
// Positive digit d → multiply by table[d-1] = g^(2d-1).
// Negative digit d → sub by table[-d-1] = g^(2(-d)-1).
// Table: odd powers only, same as windowed.
// ============================================================================

struct WNAFPlan {
    std::vector<int16_t> seq;
    int table_size = 0;

    template<int N>
    void plan(const UInt<N>& exp, int w) {
        seq.clear();
        table_size = 0;

        // Compute wNAF digits LSB first
        std::vector<int8_t> naf;
        UInt<N> e = exp;
        while (!mpn::is_zero<N>(e.d)) {
            if (e.d[0] & 1) {
                int d = static_cast<int>(e.d[0] & ((1 << w) - 1));
                if (d >= (1 << (w - 1))) d -= (1 << w);
                naf.push_back(static_cast<int16_t>(d));
                if (d > 0) mpn::sub1<N>(e.d, e.d, static_cast<uint64_t>(d));
                else       mpn::add1<N>(e.d, e.d, static_cast<uint64_t>(-d));
            } else {
                naf.push_back(0);
            }
            mpn::rshift<N>(e.d, e.d, 1);
        }

        // Strip trailing zeros (MSB side) and reverse to MSB-first
        while (!naf.empty() && naf.back() == 0) naf.pop_back();
        if (naf.empty()) return;

        // Convert to interleaved format: find nonzero digits, count zeros between them
        // naf is LSB-first; scan from back (MSB) to front (LSB)
        int top = static_cast<int>(naf.size()) - 1;

        auto digit = [](int8_t d) -> int8_t {
            int ad = d > 0 ? d : -d;
            int idx = (ad + 1) / 2;
            return d > 0 ? static_cast<int16_t>(idx) : static_cast<int16_t>(-idx);
        };

        seq.push_back(digit(naf[static_cast<size_t>(top)]));

        int pending = 0;
        for (int i = top - 1; i >= 0; --i) {
            pending++;
            if (naf[static_cast<size_t>(i)] != 0) {
                seq.push_back(static_cast<int16_t>(pending));
                seq.push_back(digit(naf[static_cast<size_t>(i)]));
                pending = 0;
            }
        }
        if (pending > 0) {
            seq.push_back(static_cast<int16_t>(pending));
            seq.push_back(0);
        }

        // Table size = max |digit| used
        for (size_t i = 0; i < seq.size(); i += (i == 0 ? 1 : 2)) {
            int v = seq[i] > 0 ? seq[i] : -seq[i];
            if (v > table_size) table_size = v;
        }
    }

    template<typename Group>
    typename Group::T exec(const typename Group::T& base) const {
        using T = typename Group::T;
        T table[32];
        table[0] = base;
        if (table_size > 1) {
            T sq = Group::dbl(base);
            for (int i = 1; i < table_size; ++i)
                table[i] = Group::add(table[i - 1], sq);
        }

        if (seq.empty()) return base;

        int8_t d0 = seq[0];
        T acc = table[(d0 > 0 ? d0 : -d0) - 1];
        // TODO: negate acc if d0 < 0 (needs Group::neg)

        for (size_t i = 1; i + 1 < seq.size(); i += 2) {
            int dbls = seq[i];
            for (int j = 0; j < dbls; ++j)
                acc = Group::dbl(acc);
            int8_t di = seq[i + 1];
            if (di > 0)      acc = Group::add(acc, table[di - 1]);
            else if (di < 0) acc = Group::sub(acc, table[-di - 1]);
        }

        return acc;
    }
};

} // namespace zfactor::fixint
