#pragma once

#include <cstdint>
#include <chrono>
#include <string>

namespace port_ai_gateway {

class TimeUtil {
public:
    static int64_t NowMicros();
    static int64_t NowMillis();
    static int64_t NowNanos();
    static int64_t NowSeconds();

    static int64_t SteadyMicros();
    static int64_t SteadyMillis();

    static std::string FormatTime(int64_t timestamp_us,
                                  const std::string& format = "%Y-%m-%d %H:%M:%S");

    static uint64_t GenUniqueId();

private:
    static std::atomic<uint64_t> unique_id_counter_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(const char* name = nullptr);
    ~ScopedTimer();

    void Reset();
    int64_t ElapsedMicros() const;
    int64_t ElapsedMillis() const;

private:
    const char* name_;
    std::chrono::steady_clock::time_point start_;
};

class FpsCounter {
public:
    explicit FpsCounter(int window_size = 30);

    void Tick();
    double GetFps() const;
    void Reset();

private:
    int window_size_;
    int count_;
    double fps_;
    std::chrono::steady_clock::time_point last_time_;
};

}
