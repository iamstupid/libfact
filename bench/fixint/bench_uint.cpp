#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <random>

#ifdef ZFACTOR_BENCH_GMP
#include <gmp.h>
#endif

#include "zfactor/fixint/uint.h"

namespace {

using zfactor::fixint::mpn::limb_t;

template<int N>
using Arr = std::array<limb_t, N>;

template<int N>
using Arr2 = std::array<limb_t, 2 * N>;

template<typename T>
inline void escape(const T& value) {
    __asm__ __volatile__("" : : "g"(&value) : "memory");
}

inline void compiler_barrier() {
    __asm__ __volatile__("" ::: "memory");
}

#define ZFACTOR_NOINLINE __attribute__((noinline))
#define ZFACTOR_USED __attribute__((used))

struct Stats {
    double ns_per_item = 0.0;
    double mops = 0.0;
};

template<typename Fn>
Stats run_bench(std::size_t items, std::size_t rounds, Fn&& fn) {
    auto start = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < rounds; ++r)
        fn();
    auto stop = std::chrono::steady_clock::now();
    double ns = std::chrono::duration<double, std::nano>(stop - start).count();
    double total_items = double(items) * double(rounds);
    return {ns / total_items, total_items / ns * 1e3};
}

template<int N>
struct Batch {
    static constexpr std::size_t count = 256;
    std::array<Arr<N>, count> a{};
    std::array<Arr<N>, count> b{};
    std::array<limb_t, count> scalar{};
    std::array<unsigned, count> shift{};

    Batch() {
        std::mt19937_64 rng(0xBADC0FFEu + N * 19u);
        for (std::size_t i = 0; i < count; ++i) {
            for (auto& limb : a[i]) limb = rng();
            for (auto& limb : b[i]) limb = rng();
            scalar[i] = rng();
            shift[i] = unsigned(rng() % 63u) + 1u;
        }
    }
};

template<int N>
Stats bench_add(const Batch<N>& batch, std::size_t rounds) {
    std::array<Arr<N>, Batch<N>::count> out{};
    volatile limb_t sink = 0;
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            sink ^= zfactor::fixint::mpn::add<N>(out[i].data(), batch.a[i].data(), batch.b[i].data());
            escape(out[i]);
        }
    });
    escape(sink);
    return stats;
}

template<int N>
Stats bench_sub(const Batch<N>& batch, std::size_t rounds) {
    std::array<Arr<N>, Batch<N>::count> out{};
    volatile limb_t sink = 0;
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            sink ^= zfactor::fixint::mpn::sub<N>(out[i].data(), batch.a[i].data(), batch.b[i].data());
            escape(out[i]);
        }
    });
    escape(sink);
    return stats;
}

template<int N>
Stats bench_addmul1(const Batch<N>& batch, std::size_t rounds) {
    auto out = batch.b;
    volatile limb_t sink = 0;
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            sink ^= zfactor::fixint::mpn::addmul1<N>(out[i].data(), batch.a[i].data(), batch.scalar[i]);
            escape(out[i]);
        }
    });
    escape(sink);
    return stats;
}

template<int N>
Stats bench_mul(const Batch<N>& batch, std::size_t rounds) {
    std::array<Arr2<N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            zfactor::fixint::mpn::mul<N>(out[i].data(), batch.a[i].data(), batch.b[i].data());
            escape(out[i]);
        }
    });
    return stats;
}

template<int N>
Stats bench_sqr(const Batch<N>& batch, std::size_t rounds) {
    std::array<Arr2<N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            zfactor::fixint::mpn::sqr<N>(out[i].data(), batch.a[i].data());
            escape(out[i]);
        }
    });
    return stats;
}

template<int N>
Stats bench_cmp(const Batch<N>& batch, std::size_t rounds) {
    volatile int sink = 0;
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            sink ^= zfactor::fixint::mpn::cmp<N>(batch.a[i].data(), batch.b[i].data());
            compiler_barrier();
        }
    });
    escape(sink);
    return stats;
}

template<int N>
Stats bench_lshift(const Batch<N>& batch, std::size_t rounds) {
    std::array<Arr<N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            zfactor::fixint::mpn::lshift<N>(out[i].data(), batch.a[i].data(), batch.shift[i]);
            escape(out[i]);
        }
    });
    return stats;
}

template<int N>
Stats bench_rshift(const Batch<N>& batch, std::size_t rounds) {
    std::array<Arr<N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            zfactor::fixint::mpn::rshift<N>(out[i].data(), batch.a[i].data(), batch.shift[i]);
            escape(out[i]);
        }
    });
    return stats;
}

#ifdef ZFACTOR_BENCH_GMP
inline mp_ptr as_gmp_ptr(limb_t* p) {
    return reinterpret_cast<mp_ptr>(p);
}

inline mp_srcptr as_gmp_srcptr(const limb_t* p) {
    return reinterpret_cast<mp_srcptr>(p);
}

template<int N>
Stats gmp_add(const Batch<N>& batch, std::size_t rounds) {
    std::array<std::array<mp_limb_t, N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            mpn_add_n(out[i].data(), as_gmp_srcptr(batch.a[i].data()), as_gmp_srcptr(batch.b[i].data()), N);
            escape(out[i]);
        }
    });
    return stats;
}

template<int N>
Stats gmp_sub(const Batch<N>& batch, std::size_t rounds) {
    std::array<std::array<mp_limb_t, N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            mpn_sub_n(out[i].data(), as_gmp_srcptr(batch.a[i].data()), as_gmp_srcptr(batch.b[i].data()), N);
            escape(out[i]);
        }
    });
    return stats;
}

template<int N>
Stats gmp_addmul1(const Batch<N>& batch, std::size_t rounds) {
    auto out = batch.b;
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            mpn_addmul_1(as_gmp_ptr(out[i].data()), as_gmp_srcptr(batch.a[i].data()), N, batch.scalar[i]);
            escape(out[i]);
        }
    });
    return stats;
}

template<int N>
Stats gmp_mul(const Batch<N>& batch, std::size_t rounds) {
    std::array<std::array<mp_limb_t, 2 * N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            mpn_mul_n(out[i].data(), as_gmp_srcptr(batch.a[i].data()), as_gmp_srcptr(batch.b[i].data()), N);
            escape(out[i]);
        }
    });
    return stats;
}

template<int N>
Stats gmp_sqr(const Batch<N>& batch, std::size_t rounds) {
    std::array<std::array<mp_limb_t, 2 * N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            mpn_sqr(out[i].data(), as_gmp_srcptr(batch.a[i].data()), N);
            escape(out[i]);
        }
    });
    return stats;
}

template<int N>
Stats gmp_cmp(const Batch<N>& batch, std::size_t rounds) {
    volatile int sink = 0;
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            sink ^= mpn_cmp(as_gmp_srcptr(batch.a[i].data()), as_gmp_srcptr(batch.b[i].data()), N);
            compiler_barrier();
        }
    });
    escape(sink);
    return stats;
}

template<int N>
Stats gmp_lshift(const Batch<N>& batch, std::size_t rounds) {
    std::array<std::array<mp_limb_t, N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            unsigned shift = batch.shift[i] % 64;
            mpn_lshift(out[i].data(), as_gmp_srcptr(batch.a[i].data()), N, shift == 0 ? 1 : shift);
            escape(out[i]);
        }
    });
    return stats;
}

template<int N>
Stats gmp_rshift(const Batch<N>& batch, std::size_t rounds) {
    std::array<std::array<mp_limb_t, N>, Batch<N>::count> out{};
    auto stats = run_bench(Batch<N>::count, rounds, [&]() {
        for (std::size_t i = 0; i < Batch<N>::count; ++i) {
            unsigned shift = batch.shift[i] % 64;
            mpn_rshift(out[i].data(), as_gmp_srcptr(batch.a[i].data()), N, shift == 0 ? 1 : shift);
            escape(out[i]);
        }
    });
    return stats;
}
#endif

inline void print_stats(const char* op, int n, const Stats& zf) {
    std::printf("%-8s N=%-2d  %8.3f ns/item  %8.2f Mop/s", op, n, zf.ns_per_item, zf.mops);
}

inline int selected_n() {
    if (const char* value = std::getenv("ZFACTOR_BENCH_ONLY_N"))
        return std::atoi(value);
    return 0;
}

inline const char* selected_op() {
    return std::getenv("ZFACTOR_BENCH_ONLY_OP");
}

inline bool want_op(const char* op) {
    const char* only_op = selected_op();
    return only_op == nullptr || std::strcmp(only_op, op) == 0;
}

template<int N>
void bench_family() {
    Batch<N> batch;
    // Use million-scale rounds so WSL scheduling noise does not dominate.
    constexpr std::size_t light = 40'000;
    constexpr std::size_t heavy = 12'000;

    Stats add{}, sub{}, addmul1{}, mul{}, sqr{}, cmp{}, lshift{}, rshift{};
    if (want_op("add")) add = bench_add<N>(batch, light);
    if (want_op("sub")) sub = bench_sub<N>(batch, light);
    if (want_op("addmul1")) addmul1 = bench_addmul1<N>(batch, light);
    if (want_op("mul")) mul = bench_mul<N>(batch, heavy);
    if (want_op("sqr")) sqr = bench_sqr<N>(batch, heavy);
    if (want_op("cmp")) cmp = bench_cmp<N>(batch, light);
    if (want_op("lshift")) lshift = bench_lshift<N>(batch, light);
    if (want_op("rshift")) rshift = bench_rshift<N>(batch, light);

#ifdef ZFACTOR_BENCH_GMP
    Stats gadd{}, gsub{}, gaddmul1{}, gmul{}, gsqr{}, gcmp{}, glshift{}, grshift{};
    if (want_op("add")) gadd = gmp_add<N>(batch, light);
    if (want_op("sub")) gsub = gmp_sub<N>(batch, light);
    if (want_op("addmul1")) gaddmul1 = gmp_addmul1<N>(batch, light);
    if (want_op("mul")) gmul = gmp_mul<N>(batch, heavy);
    if (want_op("sqr")) gsqr = gmp_sqr<N>(batch, heavy);
    if (want_op("cmp")) gcmp = gmp_cmp<N>(batch, light);
    if (want_op("lshift")) glshift = gmp_lshift<N>(batch, light);
    if (want_op("rshift")) grshift = gmp_rshift<N>(batch, light);
#endif

    auto print_one = [&](const char* op, const Stats& mine
#ifdef ZFACTOR_BENCH_GMP
        , const Stats& gmp
#endif
    ) {
        if (!want_op(op))
            return;
        print_stats(op, N, mine);
#ifdef ZFACTOR_BENCH_GMP
        std::printf("  GMP %8.3f ns/item  ratio %.3fx", gmp.ns_per_item, mine.ns_per_item / gmp.ns_per_item);
#endif
        std::printf("\n");
    };

    print_one("add", add
#ifdef ZFACTOR_BENCH_GMP
        , gadd
#endif
    );
    print_one("sub", sub
#ifdef ZFACTOR_BENCH_GMP
        , gsub
#endif
    );
    print_one("addmul1", addmul1
#ifdef ZFACTOR_BENCH_GMP
        , gaddmul1
#endif
    );
    print_one("mul", mul
#ifdef ZFACTOR_BENCH_GMP
        , gmul
#endif
    );
    print_one("sqr", sqr
#ifdef ZFACTOR_BENCH_GMP
        , gsqr
#endif
    );
    print_one("cmp", cmp
#ifdef ZFACTOR_BENCH_GMP
        , gcmp
#endif
    );
    print_one("lshift", lshift
#ifdef ZFACTOR_BENCH_GMP
        , glshift
#endif
    );
    print_one("rshift", rshift
#ifdef ZFACTOR_BENCH_GMP
        , grshift
#endif
    );
    std::printf("\n");
}

template<int N>
void maybe_bench_family() {
    int only_n = selected_n();
    if (only_n == 0 || only_n == N)
        bench_family<N>();
}

extern "C" ZFACTOR_USED ZFACTOR_NOINLINE
std::uint64_t zfactor_codegen_add4(std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b) {
    return zfactor::fixint::mpn::add<4>(r, a, b);
}

} // namespace

int main() {
#ifdef ZFACTOR_BENCH_GMP
    static_assert(sizeof(mp_limb_t) == sizeof(limb_t));
#endif

    std::puts("== zfactor fixint benchmark ==");
    maybe_bench_family<1>();
    maybe_bench_family<2>();
    maybe_bench_family<3>();
    maybe_bench_family<4>();
    maybe_bench_family<6>();
    maybe_bench_family<8>();
    maybe_bench_family<12>();
    maybe_bench_family<16>();
    return 0;
}
