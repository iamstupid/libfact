#pragma once

#include <array>
#include <cctype>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <iosfwd>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>

#include "mpn.h"

namespace zfactor::fixint {

template<int N>
struct UInt {
    static_assert(N >= 1, "UInt<N> requires N >= 1");

    using limb_type = mpn::limb_t;
    static constexpr int limbs = N;

    limb_type d[N] = {};

    constexpr UInt() = default;
    constexpr UInt(std::uint64_t v) : d{} { d[0] = v; }

    constexpr limb_type* data() { return d; }
    constexpr const limb_type* data() const { return d; }

    constexpr limb_type& operator[](int i) { return d[i]; }
    constexpr const limb_type& operator[](int i) const { return d[i]; }

    explicit constexpr operator bool() const { return !mpn::is_zero<N>(d); }

    friend bool operator==(const UInt& a, const UInt& b) {
        return mpn::cmp<N>(a.d, b.d) == 0;
    }

    friend auto operator<=>(const UInt& a, const UInt& b) {
        int c = mpn::cmp<N>(a.d, b.d);
        if (c < 0) return std::strong_ordering::less;
        if (c > 0) return std::strong_ordering::greater;
        return std::strong_ordering::equal;
    }

    UInt& operator+=(const UInt& other) {
        (void)mpn::add<N>(d, d, other.d);
        return *this;
    }

    UInt& operator-=(const UInt& other) {
        (void)mpn::sub<N>(d, d, other.d);
        return *this;
    }

    UInt& operator&=(const UInt& other) {
        mpn::bitand_<N>(d, d, other.d);
        return *this;
    }

    UInt& operator|=(const UInt& other) {
        mpn::bitor_<N>(d, d, other.d);
        return *this;
    }

    UInt& operator^=(const UInt& other) {
        mpn::bitxor_<N>(d, d, other.d);
        return *this;
    }

    UInt& operator<<=(unsigned bits) {
        mpn::lshift<N>(d, d, bits);
        return *this;
    }

    UInt& operator>>=(unsigned bits) {
        mpn::rshift<N>(d, d, bits);
        return *this;
    }

    friend UInt operator+(UInt a, const UInt& b) { return a += b; }
    friend UInt operator-(UInt a, const UInt& b) { return a -= b; }
    friend UInt operator&(UInt a, const UInt& b) { return a &= b; }
    friend UInt operator|(UInt a, const UInt& b) { return a |= b; }
    friend UInt operator^(UInt a, const UInt& b) { return a ^= b; }
    friend UInt operator<<(UInt a, unsigned bits) { return a <<= bits; }
    friend UInt operator>>(UInt a, unsigned bits) { return a >>= bits; }
    friend UInt operator~(UInt a) {
        mpn::bitnot_<N>(a.d, a.d);
        return a;
    }

    [[nodiscard]] bool is_zero() const { return mpn::is_zero<N>(d); }
    [[nodiscard]] unsigned bit_length() const { return mpn::bit_length<N>(d); }

    [[nodiscard]] std::string to_hex(bool prefix = true) const {
        static constexpr char hexdig[] = "0123456789abcdef";
        std::string out;
        if (prefix)
            out = "0x";

        int top = N - 1;
        while (top > 0 && d[top] == 0)
            --top;

        bool started = false;
        for (int i = top; i >= 0; --i) {
            for (int nib = 15; nib >= 0; --nib) {
                unsigned val = unsigned((d[i] >> (nib * 4)) & 0xF);
                if (!started) {
                    if (val == 0 && (i != 0 || nib != 0))
                        continue;
                    started = true;
                }
                out.push_back(hexdig[val]);
            }
        }
        if (!started)
            out.push_back('0');
        return out;
    }

    static UInt from_hex(std::string_view text) {
        UInt out;
        if (text.starts_with("0x") || text.starts_with("0X"))
            text.remove_prefix(2);
        for (char ch : text) {
            unsigned value = 0;
            if (ch >= '0' && ch <= '9')
                value = unsigned(ch - '0');
            else if (ch >= 'a' && ch <= 'f')
                value = unsigned(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F')
                value = unsigned(ch - 'A' + 10);
            else
                continue;
            mpn::lshift<N>(out.d, out.d, 4);
            (void)mpn::add1<N>(out.d, out.d, value);
        }
        return out;
    }
};

template<int N>
struct UIntWide {
    UInt<N> lo{};
    UInt<N> hi{};

    using limb_type = mpn::limb_t;
    static constexpr int limbs = 2 * N;

    constexpr limb_type* data() { return lo.d; }
    constexpr const limb_type* data() const { return lo.d; }
};

template<int N>
inline UIntWide<N> operator*(const UInt<N>& a, const UInt<N>& b) {
    UIntWide<N> out;
    mpn::mul<N>(out.data(), a.data(), b.data());
    return out;
}

template<int N>
inline std::ostream& operator<<(std::ostream& os, const UInt<N>& value) {
    return os << value.to_hex();
}

template<int N>
inline std::istream& operator>>(std::istream& is, UInt<N>& value) {
    std::string token;
    is >> token;
    if (!is)
        return is;
    value = UInt<N>::from_hex(token);
    return is;
}

template<int N>
inline bool is_standard_layout_v = std::is_standard_layout_v<UInt<N>>;

static_assert(std::is_standard_layout_v<UInt<1>>);
static_assert(std::is_standard_layout_v<UIntWide<1>>);
static_assert(sizeof(UIntWide<2>) == sizeof(mpn::limb_t) * 4);
static_assert(offsetof(UIntWide<2>, hi) == sizeof(UInt<2>));

} // namespace zfactor::fixint
