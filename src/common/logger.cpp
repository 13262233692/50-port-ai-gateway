#include "common/logger.h"
#include <thread>

namespace port_ai_gateway {

Logger& Logger::Instance() {
    static Logger instance;
    return instance;
}

void Logger::SetLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

LogLevel Logger::GetLevel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return level_;
}

void Logger::Log(LogLevel level, const std::string& file, int line,
                  const std::string& func, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level < level_) {
        return;
    }

    std::ostream& out = (level >= LogLevel::ERROR) ? std::cerr : std::cout;

    out << "[" << GetTimestamp() << "]"
        << "[" << LevelToString(level) << "]"
        << "[" << std::this_thread::get_id() << "]"
        << "[" << file << ":" << line << "]"
        << "[" << func << "] "
        << message << std::endl;

    if (level == LogLevel::FATAL) {
        std::abort();
    }
}

const char* Logger::LevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

std::string Logger::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()) % 1000000;

    struct tm tm_time;
#ifdef _WIN32
    localtime_s(&tm_time, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_time);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_time, "%Y-%m-%d %H:%M:%S")
        << "." << std::setw(6) << std::setfill('0') << us.count();
    return oss.str();
}

}
