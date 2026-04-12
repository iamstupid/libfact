// Factor a 220-bit semiprime with multithreaded ECM.
#include "zfactor/eecm.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

using namespace zfactor::fixint;
using namespace zfactor::ecm;
using Clock = std::chrono::steady_clock;

std::atomic<bool> found_flag{false};
std::atomic<int> total_curves{0};

template<int N>
void worker(UInt<N> n, int thread_id, int nthreads, UInt<N>* result) {
    MontCtx<N> ctx;
    ctx.init(n);

    int fb = 110;  // 220/2
    const auto& row = schedule_for_bits(fb);

    // Each thread gets its own curve parameter range
    uint64_t k = 2 + thread_id;
    int curves_done = 0;

    while (!found_flag.load(std::memory_order_relaxed)) {
        auto r = ecm_one_curve<N>(k, row.B1, row.B2, ctx);
        k += nthreads;
        curves_done++;
        total_curves.fetch_add(1, std::memory_order_relaxed);

        if (r.factor_found) {
            found_flag.store(true, std::memory_order_relaxed);
            *result = r.factor;
            std::printf("  Thread %d found factor at curve %d (k=%lu)\n",
                        thread_id, curves_done, k - nthreads);
            return;
        }
    }
}

int main() {
    // 240bit = 664613997892457936451903530140173457 * 664613997892457936451903530140183751
    UInt<4> n{};
    uint64_t limbs[] = {0x0000000000CC78B7ULL, 0xAC00000000000000ULL,
                        0x0000000000000018ULL, 0x0000400000000000ULL};
    std::memcpy(n.d, limbs, 32);

    int fb = 120;
    const auto& row = schedule_for_bits(fb);
    int nthreads = (int)std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;

    std::printf("=== 240-bit ECM challenge ===\n");
    std::printf("n = 441711766194596082395824375185738024360892351747557999975794280992897207\n");
    std::printf("B1=%lu  B2=%lu  expected_curves=%u\n", row.B1, row.B2, row.avg_curves);
    std::printf("Using %d threads\n\n", nthreads);

    UInt<4> result{};
    auto t0 = Clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < nthreads; ++i)
        threads.emplace_back(worker<4>, n, i, nthreads, &result);

    for (auto& t : threads)
        t.join();

    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (found_flag.load()) {
        std::printf("\nFactor: %llu %llu %llu %llu\n",
                    (unsigned long long)result.d[0],
                    (unsigned long long)result.d[1],
                    (unsigned long long)result.d[2],
                    (unsigned long long)result.d[3]);
        std::printf("Total curves: %d\n", total_curves.load());
        std::printf("Wall time: %.1f ms (%.2f s)\n", ms, ms / 1000);
        std::printf("Throughput: %.0f curves/s\n", total_curves.load() / (ms / 1000));
    } else {
        std::printf("\nNot found after %d curves (%.1f s)\n",
                    total_curves.load(), ms / 1000);
    }
    return 0;
}
