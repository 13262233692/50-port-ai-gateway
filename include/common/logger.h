#pragma once

#include <string>
#include <sstream>
#include <iostream>
#include <mutex>
#include <chrono>
#include <iomanip>

namespace port_ai_gateway {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

class Logger {
public:
    static Logger& Instance();

    void SetLevel(LogLevel level);
    LogLevel GetLevel() const;

    void Log(LogLevel level, const std::string& file, int line,
             const std::string& func, const std::string& message);

private:
    Logger() = default;
    ~Logger() = default;

    static const char* LevelToString(LogLevel level);
    static std::string GetTimestamp();

    LogLevel level_ = LogLevel::INFO;
    mutable std::mutex mutex_;
};

class LogStream {
public:
    LogStream(LogLevel level, const char* file, int line, const char* func)
        : level_(level), file_(file), line_(line), func_(func) {}

    ~LogStream() {
        Logger::Instance().Log(level_, file_, line_, func_, stream_.str());
    }

    std::ostringstream& Stream() { return stream_; }

private:
    LogLevel level_;
    std::string file_;
    int line_;
    std::string func_;
    std::ostringstream stream_;
};

#define LOG_TRACE ::port_ai_gateway::LogStream(::port_ai_gateway::LogLevel::TRACE, \
    __FILE__, __LINE__, __func__).Stream()
#define LOG_DEBUG ::port_ai_gateway::LogStream(::port_ai_gateway::LogLevel::DEBUG, \
    __FILE__, __LINE__, __func__).Stream()
#define LOG_INFO  ::port_ai_gateway::LogStream(::port_ai_gateway::LogLevel::INFO, \
    __FILE__, __LINE__, __func__).Stream()
#define LOG_WARN  ::port_ai_gateway::LogStream(::port_ai_gateway::LogLevel::WARN, \
    __FILE__, __LINE__, __func__).Stream()
#define LOG_ERROR ::port_ai_gateway::LogStream(::port_ai_gateway::LogLevel::ERROR, \
    __FILE__, __LINE__, __func__).Stream()
#define LOG_FATAL ::port_ai_gateway::LogStream(::port_ai_gateway::LogLevel::FATAL, \
    __FILE__, __LINE__, __func__).Stream()

}
