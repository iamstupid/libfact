#pragma once

#include <chrono>
#include <cstdint>

namespace zfactor {

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_;

public:
    Timer() : start_(Clock::now()) {}
    void reset() { start_ = Clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }

    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
    }

    uint64_t elapsed_ns() const {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count());
    }
};

} // namespace zfactor
