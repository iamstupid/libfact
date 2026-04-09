#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "zfactor/fixint/uint.h"

namespace {

using zfactor::fixint::mpn::limb_t;

template<std::size_t N>
using Arr = std::array<limb_t, N>;

inline limb_t manual_addcarry(uint8_t c, limb_t a, limb_t b, limb_t* out) {
    limb_t sum = a + b;
    limb_t c1 = sum < a;
    limb_t sum2 = sum + c;
    limb_t c2 = sum2 < sum;
    *out = sum2;
    return c1 | c2;
}

inline limb_t manual_subborrow(uint8_t c, limb_t a, limb_t b, limb_t* out) {
    limb_t diff = a - b;
    limb_t b1 = diff > a;
    limb_t diff2 = diff - c;
    limb_t b2 = diff2 > diff;
    *out = diff2;
    return b1 | b2;
}

inline limb_t manual_umul128(limb_t a, limb_t b, limb_t* hi) {
    limb_t a0 = uint32_t(a);
    limb_t a1 = a >> 32;
    limb_t b0 = uint32_t(b);
    limb_t b1 = b >> 32;

    limb_t p00 = a0 * b0;
    limb_t p01 = a0 * b1;
    limb_t p10 = a1 * b0;
    limb_t p11 = a1 * b1;

    limb_t middle = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
    return (p00 & 0xFFFFFFFFull) | (middle << 32);
}

template<std::size_t N>
std::string to_hex(const Arr<N>& a) {
    static constexpr char hexdig[] = "0123456789abcdef";
    std::string out = "0x";
    bool started = false;
    for (int i = int(N) - 1; i >= 0; --i) {
        for (int nib = 15; nib >= 0; --nib) {
            unsigned v = unsigned((a[std::size_t(i)] >> (nib * 4)) & 0xF);
            if (!started) {
                if (v == 0 && (i != 0 || nib != 0))
                    continue;
                started = true;
            }
            out.push_back(hexdig[v]);
        }
    }
    if (!started)
        out.push_back('0');
    return out;
}

template<std::size_t N>
void require_eq(const Arr<N>& actual, const Arr<N>& expected, const char* label) {
    if (actual != expected) {
        auto msg = std::string(label) + " actual=" + to_hex(actual) + " expected=" + to_hex(expected);
        std::fprintf(stderr, "%s\n", msg.c_str());
        CHECK_MESSAGE(actual == expected, msg);
        return;
    }
    CHECK(true);
}

template<int N>
Arr<N> ref_lshift(const Arr<N>& a, unsigned bits) {
    Arr<N> out{};
    if (bits >= unsigned(N * 64))
        return out;
    unsigned limb_shift = bits / 64;
    unsigned bit_shift = bits % 64;
    for (int i = N - 1; i >= 0; --i) {
        limb_t v = 0;
        if (i >= int(limb_shift)) {
            v = a[std::size_t(i - limb_shift)] << bit_shift;
            if (bit_shift != 0 && i > int(limb_shift))
                v |= a[std::size_t(i - limb_shift - 1)] >> (64 - bit_shift);
        }
        out[std::size_t(i)] = v;
    }
    return out;
}

template<int N>
Arr<N> ref_rshift(const Arr<N>& a, unsigned bits) {
    Arr<N> out{};
    if (bits >= unsigned(N * 64))
        return out;
    unsigned limb_shift = bits / 64;
    unsigned bit_shift = bits % 64;
    for (int i = 0; i < N; ++i) {
        limb_t v = 0;
        if (i + int(limb_shift) < N) {
            v = a[std::size_t(i + limb_shift)] >> bit_shift;
            if (bit_shift != 0 && i + int(limb_shift) + 1 < N)
                v |= a[std::size_t(i + limb_shift + 1)] << (64 - bit_shift);
        }
        out[std::size_t(i)] = v;
    }
    return out;
}

template<int N>
Arr<N> clear_high_bits(Arr<N> a, unsigned bits) {
    unsigned total = N * 64;
    if (bits >= total) {
        a.fill(0);
        return a;
    }
    for (unsigned bit = total - bits; bit < total; ++bit)
        a[bit / 64] &= ~(limb_t(1) << (bit % 64));
    return a;
}

template<int N>
Arr<N> clear_low_bits(Arr<N> a, unsigned bits) {
    unsigned total = N * 64;
    unsigned lim = std::min(bits, total);
    for (unsigned bit = 0; bit < lim; ++bit)
        a[bit / 64] &= ~(limb_t(1) << (bit % 64));
    return a;
}

template<int N>
std::pair<Arr<N>, limb_t> ref_add(const Arr<N>& a, const Arr<N>& b) {
    Arr<N> out{};
    limb_t cy = 0;
    for (int i = 0; i < N; ++i)
        cy = manual_addcarry((uint8_t)cy, a[std::size_t(i)], b[std::size_t(i)], out.data() + i);
    return {out, cy};
}

template<int N>
std::pair<Arr<N>, limb_t> ref_sub(const Arr<N>& a, const Arr<N>& b) {
    Arr<N> out{};
    limb_t bw = 0;
    for (int i = 0; i < N; ++i)
        bw = manual_subborrow((uint8_t)bw, a[std::size_t(i)], b[std::size_t(i)], out.data() + i);
    return {out, bw};
}

template<int N>
std::pair<Arr<N>, limb_t> ref_addmul1(Arr<N> r, const Arr<N>& a, limb_t b) {
    limb_t cy = 0;
    for (int i = 0; i < N; ++i) {
        limb_t hi = 0;
        limb_t lo = manual_umul128(a[std::size_t(i)], b, &hi);
        limb_t sum = 0;
        uint8_t carry = (uint8_t)manual_addcarry(0, lo, r[std::size_t(i)], &sum);
        uint8_t carry2 = (uint8_t)manual_addcarry(0, sum, cy, &sum);
        r[std::size_t(i)] = sum;
        cy = hi + carry + carry2;
    }
    return {r, cy};
}

template<int N>
limb_t ref_addmul(Arr<2 * N>& r, const Arr<N>& a, const Arr<N>& b) {
    limb_t overflow = 0;
    for (int j = 0; j < N; ++j) {
        limb_t cy = 0;
        for (int i = 0; i < N; ++i) {
            limb_t hi = 0;
            limb_t lo = manual_umul128(a[std::size_t(i)], b[std::size_t(j)], &hi);
            limb_t sum = 0;
            uint8_t carry = (uint8_t)manual_addcarry(0, lo, r[std::size_t(i + j)], &sum);
            uint8_t carry2 = (uint8_t)manual_addcarry(0, sum, cy, &sum);
            r[std::size_t(i + j)] = sum;
            cy = hi + carry + carry2;
        }
        int k = j + N;
        limb_t carry = manual_addcarry(0, r[std::size_t(k)], cy, r.data() + k);
        ++k;
        while (carry != 0 && k < 2 * N) {
            carry = manual_addcarry((uint8_t)carry, r[std::size_t(k)], 0, r.data() + k);
            ++k;
        }
        overflow |= carry;
    }
    return overflow;
}

template<int N>
Arr<2 * N> ref_mul(const Arr<N>& a, const Arr<N>& b) {
    Arr<2 * N> out{};
    (void)ref_addmul<N>(out, a, b);
    return out;
}

template<int N>
Arr<2 * N> ref_sqr(const Arr<N>& a) {
    return ref_mul<N>(a, a);
}

template<int N>
unsigned ref_clz(const Arr<N>& a) {
    for (int i = N - 1; i >= 0; --i) {
        if (a[std::size_t(i)] != 0)
            return unsigned((N - 1 - i) * 64 + std::countl_zero(a[std::size_t(i)]));
    }
    return unsigned(N * 64);
}

template<int N>
unsigned ref_ctz(const Arr<N>& a) {
    for (int i = 0; i < N; ++i) {
        if (a[std::size_t(i)] != 0)
            return unsigned(i * 64 + std::countr_zero(a[std::size_t(i)]));
    }
    return unsigned(N * 64);
}

template<int N>
unsigned ref_bit_length(const Arr<N>& a) {
    return unsigned(N * 64) - ref_clz<N>(a);
}

template<int N>
int ref_cmp(const Arr<N>& a, const Arr<N>& b) {
    for (int i = N - 1; i >= 0; --i) {
        if (a[std::size_t(i)] < b[std::size_t(i)]) return -1;
        if (a[std::size_t(i)] > b[std::size_t(i)]) return 1;
    }
    return 0;
}

template<int N>
std::vector<Arr<N>> build_cases() {
    std::vector<Arr<N>> cases;
    Arr<N> zero{};
    Arr<N> ones{};
    ones.fill(~limb_t(0));
    Arr<N> low_one{};
    low_one[0] = 1;
    Arr<N> high_one{};
    high_one[N - 1] = limb_t(1) << 63;
    Arr<N> alt_a{};
    Arr<N> alt_b{};
    Arr<N> ramp{};
    for (int i = 0; i < N; ++i) {
        alt_a[std::size_t(i)] = 0xAAAAAAAAAAAAAAAAull;
        alt_b[std::size_t(i)] = 0x5555555555555555ull;
        ramp[std::size_t(i)] = 0x0102030405060708ull * limb_t(i + 1);
    }
    cases.push_back(zero);
    cases.push_back(ones);
    cases.push_back(low_one);
    cases.push_back(high_one);
    cases.push_back(alt_a);
    cases.push_back(alt_b);
    cases.push_back(ramp);

    std::mt19937_64 rng(0xC0FFEEu + N * 17u);
    for (int i = 0; i < 64; ++i) {
        Arr<N> x{};
        for (auto& limb : x)
            limb = rng();
        cases.push_back(x);
    }
    return cases;
}

template<int N>
zfactor::fixint::UInt<N> make_uint(const Arr<N>& a) {
    zfactor::fixint::UInt<N> out;
    std::copy(a.begin(), a.end(), out.d);
    return out;
}

template<int N>
void run_suite() {
    using namespace zfactor::fixint;
    auto cases = build_cases<N>();

    {
        limb_t hi = 0;
        limb_t lo = mpn::umul128(0xFEDCBA9876543210ull, 0x0123456789ABCDEFull, &hi);
        limb_t ref_hi = 0;
        limb_t ref_lo = manual_umul128(0xFEDCBA9876543210ull, 0x0123456789ABCDEFull, &ref_hi);
        CHECK(lo == ref_lo);
        CHECK(hi == ref_hi);
    }

    for (const auto& a : cases) {
        Arr<N> tmp{};
        mpn::set_zero<N>(tmp.data());
        CHECK(mpn::is_zero<N>(tmp.data()));
        mpn::copy<N>(tmp.data(), a.data());
        require_eq(tmp, a, "copy");
        CHECK(mpn::is_zero<N>(a.data()) == std::all_of(a.begin(), a.end(), [](limb_t v) { return v == 0; }));
        CHECK(mpn::clz<N>(a.data()) == ref_clz<N>(a));
        CHECK(mpn::ctz<N>(a.data()) == ref_ctz<N>(a));
        CHECK(mpn::bit_length<N>(a.data()) == ref_bit_length<N>(a));

        for (unsigned bit : {0u, 1u, 7u, 8u, 31u, 32u, 63u, unsigned(N * 64 / 2), unsigned(N * 64 - 1)}) {
            Arr<N> bits = a;
            bool before = mpn::test_bit<N>(bits.data(), bit);
            mpn::set_bit<N>(bits.data(), bit);
            CHECK(mpn::test_bit<N>(bits.data(), bit));
            mpn::clear_bit<N>(bits.data(), bit);
            CHECK(!mpn::test_bit<N>(bits.data(), bit));
            if (before)
                mpn::set_bit<N>(bits.data(), bit);
            CHECK(mpn::test_bit<N>(bits.data(), bit) == before);
        }

        std::vector<unsigned> shifts = {0, 1, 2, 7, 8, 15, 31, 32, 63, 64, 65};
        shifts.push_back(unsigned(N * 64 - 1));
        std::sort(shifts.begin(), shifts.end());
        shifts.erase(std::unique(shifts.begin(), shifts.end()), shifts.end());

        for (unsigned bits : shifts) {
            Arr<N> l{};
            Arr<N> r{};
            mpn::lshift<N>(l.data(), a.data(), bits);
            mpn::rshift<N>(r.data(), a.data(), bits);
            require_eq(l, ref_lshift<N>(a, bits), "lshift");
            require_eq(r, ref_rshift<N>(a, bits), "rshift");

            Arr<N> round = clear_high_bits<N>(a, bits);
            Arr<N> tmp2{};
            mpn::lshift<N>(tmp2.data(), round.data(), bits);
            mpn::rshift<N>(tmp2.data(), tmp2.data(), bits);
            require_eq(tmp2, round, "lshift/rshift roundtrip");

            round = clear_low_bits<N>(a, bits);
            mpn::rshift<N>(tmp2.data(), round.data(), bits);
            mpn::lshift<N>(tmp2.data(), tmp2.data(), bits);
            require_eq(tmp2, round, "rshift/lshift roundtrip");
        }

        Arr<N> one_left{};
        Arr<N> one_right{};
        limb_t lc = mpn::lshift1<N>(one_left.data(), a.data());
        limb_t rc = mpn::rshift1<N>(one_right.data(), a.data());
        CHECK(lc == (a[N - 1] >> 63));
        CHECK(rc == (a[0] & 1));
        require_eq(one_left, ref_lshift<N>(a, 1), "lshift1");
        require_eq(one_right, ref_rshift<N>(a, 1), "rshift1");

        auto ua = make_uint<N>(a);
        auto hex = ua.to_hex();
        CHECK(zfactor::fixint::UInt<N>::from_hex(hex) == ua);
    }

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& a = cases[i];
        const auto& b = cases[(i * 7 + 3) % cases.size()];

        Arr<N> add_out{};
        Arr<N> sub_out{};
        Arr<N> and_out{};
        Arr<N> or_out{};
        Arr<N> xor_out{};
        Arr<N> not_out{};

        auto [ref_add_out, ref_add_cy] = ref_add<N>(a, b);
        auto [ref_sub_out, ref_sub_bw] = ref_sub<N>(a, b);

        limb_t add_cy = mpn::add<N>(add_out.data(), a.data(), b.data());
        limb_t sub_bw = mpn::sub<N>(sub_out.data(), a.data(), b.data());
        mpn::bitand_<N>(and_out.data(), a.data(), b.data());
        mpn::bitor_<N>(or_out.data(), a.data(), b.data());
        mpn::bitxor_<N>(xor_out.data(), a.data(), b.data());
        mpn::bitnot_<N>(not_out.data(), a.data());

        require_eq(add_out, ref_add_out, "add");
        require_eq(sub_out, ref_sub_out, "sub");
        CHECK(add_cy == ref_add_cy);
        CHECK(sub_bw == ref_sub_bw);
        CHECK(mpn::cmp<N>(a.data(), b.data()) == ref_cmp<N>(a, b));

        for (int limb = 0; limb < N; ++limb) {
            CHECK(and_out[std::size_t(limb)] == (a[std::size_t(limb)] & b[std::size_t(limb)]));
            CHECK(or_out[std::size_t(limb)] == (a[std::size_t(limb)] | b[std::size_t(limb)]));
            CHECK(xor_out[std::size_t(limb)] == (a[std::size_t(limb)] ^ b[std::size_t(limb)]));
            CHECK(not_out[std::size_t(limb)] == ~a[std::size_t(limb)]);
        }

        Arr<N> restore{};
        (void)mpn::sub<N>(restore.data(), add_out.data(), b.data());
        require_eq(restore, a, "(a+b)-b");

        Arr<N> add1_rhs{};
        add1_rhs[0] = b[0];
        Arr<N> add1_out{};
        Arr<N> sub1_out{};
        auto [ref_add1_out, ref_add1_cy] = ref_add<N>(a, add1_rhs);
        auto [ref_sub1_out, ref_sub1_bw] = ref_sub<N>(a, add1_rhs);
        limb_t add1_cy = mpn::add1<N>(add1_out.data(), a.data(), b[0]);
        limb_t sub1_bw = mpn::sub1<N>(sub1_out.data(), a.data(), b[0]);
        require_eq(add1_out, ref_add1_out, "add1");
        require_eq(sub1_out, ref_sub1_out, "sub1");
        CHECK(add1_cy == ref_add1_cy);
        CHECK(sub1_bw == ref_sub1_bw);

        Arr<N> accum = b;
        Arr<N> addmul1_out = accum;
        limb_t scalar = b[0] ^ 0x9E3779B97F4A7C15ull;
        auto [ref_addmul1_out, ref_addmul1_cy] = ref_addmul1<N>(accum, a, scalar);
        limb_t addmul1_cy = mpn::addmul1<N>(addmul1_out.data(), a.data(), scalar);
        require_eq(addmul1_out, ref_addmul1_out, "addmul1");
        CHECK(addmul1_cy == ref_addmul1_cy);

        Arr<2 * N> mul_out{};
        Arr<2 * N> sqr_out{};
        Arr<2 * N> addmul_acc{};
        Arr<2 * N> ref_acc{};
        for (int j = 0; j < 2 * N; ++j) {
            addmul_acc[std::size_t(j)] = limb_t(0x1111111111111111ull * limb_t(j + 1)) ^ a[std::size_t(j % N)];
            ref_acc[std::size_t(j)] = addmul_acc[std::size_t(j)];
        }
        mpn::mul<N>(mul_out.data(), a.data(), b.data());
        mpn::sqr<N>(sqr_out.data(), a.data());
        limb_t addmul_overflow = mpn::addmul<N>(addmul_acc.data(), a.data(), b.data());

        auto ref_mul_out = ref_mul<N>(a, b);
        auto ref_sqr_out = ref_sqr<N>(a);
        auto ref_addmul_overflow = ref_addmul<N>(ref_acc, a, b);
        require_eq(mul_out, ref_mul_out, "mul");
        require_eq(sqr_out, ref_sqr_out, "sqr");
        require_eq(sqr_out, ref_mul<N>(a, a), "sqr==mul");
        require_eq(addmul_acc, ref_acc, "addmul");
        CHECK(addmul_overflow == ref_addmul_overflow);

        auto ua = make_uint<N>(a);
        auto ub = make_uint<N>(b);
        CHECK(zfactor::fixint::UInt<N>::from_hex(ua.to_hex()) == ua);
        CHECK((ua + ub) == make_uint<N>(ref_add_out));
        CHECK((ua - ub) == make_uint<N>(ref_sub_out));
        CHECK((ua & ub) == make_uint<N>(and_out));
        CHECK((ua | ub) == make_uint<N>(or_out));
        CHECK((ua ^ ub) == make_uint<N>(xor_out));
        CHECK((~ua) == make_uint<N>(not_out));
        CHECK((ua << 13) == make_uint<N>(ref_lshift<N>(a, 13)));
        CHECK((ua >> 11) == make_uint<N>(ref_rshift<N>(a, 11)));

        auto wide = ua * ub;
        CHECK(std::equal(wide.data(), wide.data() + 2 * N, ref_mul_out.begin()));
    }
}

} // namespace

TEST_CASE("single limb primitives") {
    using namespace zfactor::fixint::mpn;
    std::mt19937_64 rng(123456789u);
    for (int i = 0; i < 1000; ++i) {
        limb_t a = rng();
        limb_t b = rng();
        uint8_t c = uint8_t(rng() & 1u);
        limb_t out = 0;
        limb_t hi = 0;
        limb_t lo = umul128(a, b, &hi);
        limb_t ref_hi = 0;
        limb_t ref_lo = manual_umul128(a, b, &ref_hi);
        CHECK(lo == ref_lo);
        CHECK(hi == ref_hi);

        uint8_t cy = addcarry(c, a, b, &out);
        limb_t ref_out = 0;
        CHECK(out == out);
        CHECK(cy == manual_addcarry(c, a, b, &ref_out));
        CHECK(out == ref_out);

        uint8_t bw = subborrow(c, a, b, &out);
        CHECK(bw == manual_subborrow(c, a, b, &ref_out));
        CHECK(out == ref_out);
    }
}

#define ZFACTOR_TEST_N_LIST(X) \
    X(1)                       \
    X(2)                       \
    X(3)                       \
    X(4)                       \
    X(5)                       \
    X(6)                       \
    X(7)                       \
    X(8)                       \
    X(9)                       \
    X(12)                      \
    X(16)                      \
    X(17)                      \
    X(24)

#define ZFACTOR_MAKE_TEST(N) \
    TEST_CASE("mpn and UInt suite N=" #N) { run_suite<N>(); }

ZFACTOR_TEST_N_LIST(ZFACTOR_MAKE_TEST)

#undef ZFACTOR_MAKE_TEST
#undef ZFACTOR_TEST_N_LIST
