#include "common/time_util.h"
#include <ctime>
#include <iomanip>
#include <sstream>
#include <atomic>

namespace port_ai_gateway {

std::atomic<uint64_t> TimeUtil::unique_id_counter_{0};

int64_t TimeUtil::NowMicros() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

int64_t TimeUtil::NowMillis() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

int64_t TimeUtil::NowNanos() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

int64_t TimeUtil::NowSeconds() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

int64_t TimeUtil::SteadyMicros() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

int64_t TimeUtil::SteadyMillis() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

std::string TimeUtil::FormatTime(int64_t timestamp_us, const std::string& format) {
    time_t sec = static_cast<time_t>(timestamp_us / 1000000);
    int64_t us = timestamp_us % 1000000;

    struct tm tm_time;
#ifdef _WIN32
    localtime_s(&tm_time, &sec);
#else
    localtime_r(&sec, &tm_time);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_time, format.c_str());
    oss << "." << std::setw(6) << std::setfill('0') << us;
    return oss.str();
}

uint64_t TimeUtil::GenUniqueId() {
    return ++unique_id_counter_;
}

ScopedTimer::ScopedTimer(const char* name)
    : name_(name), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() = default;

void ScopedTimer::Reset() {
    start_ = std::chrono::steady_clock::now();
}

int64_t ScopedTimer::ElapsedMicros() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count();
}

int64_t ScopedTimer::ElapsedMillis() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_).count();
}

FpsCounter::FpsCounter(int window_size)
    : window_size_(window_size > 0 ? window_size : 30)
    , count_(0)
    , fps_(0.0)
    , last_time_(std::chrono::steady_clock::now()) {}

void FpsCounter::Tick() {
    count_++;
    if (count_ >= window_size_) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_time_).count();
        if (elapsed > 0) {
            fps_ = static_cast<double>(count_) / elapsed;
        }
        count_ = 0;
        last_time_ = now;
    }
}

double FpsCounter::GetFps() const {
    return fps_;
}

void FpsCounter::Reset() {
    count_ = 0;
    fps_ = 0.0;
    last_time_ = std::chrono::steady_clock::now();
}

}
