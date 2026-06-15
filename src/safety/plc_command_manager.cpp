#include "safety/plc_command_manager.h"
#include "common/logger.h"
#include "common/time_util.h"

#include <algorithm>

namespace port_ai_gateway {

PlcCommandManager::PlcCommandManager()
    : running_{false}
    , emergency_active_{false}
    , next_command_id_(1) {
}

PlcCommandManager::~PlcCommandManager() {
    Stop();
    Release();
}

bool PlcCommandManager::Init(const PlcManagerConfig& config) {
    config_ = config;
    stats_ = {};
    next_command_id_ = 1;
    LOG_INFO << "PlcCommandManager initialized: "
             << "estop=" << config_.enable_emergency_stop
             << ", speed_limit=" << config_.enable_speed_limit
             << ", damping=" << config_.enable_swing_damping
             << ", min_interval=" << config_.min_interval_between_same_type_ms << "ms";
    return true;
}

void PlcCommandManager::Release() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!priority_queue_.empty()) {
        priority_queue_.pop();
    }
    command_history_.clear();
    last_issue_time_us_.clear();
    stats_ = {};
    emergency_active_.store(false);
}

bool PlcCommandManager::Start() {
    if (running_.load()) {
        LOG_WARN << "PlcCommandManager already running";
        return true;
    }
    running_.store(true);
    LOG_INFO << "PlcCommandManager started";
    return true;
}

void PlcCommandManager::Stop() {
    running_.store(false);
    LOG_INFO << "PlcCommandManager stopped";
}

bool PlcCommandManager::ShouldThrottle(const PlcCommand& cmd) const {
    auto it = last_issue_time_us_.find(cmd.type);
    if (it == last_issue_time_us_.end()) return false;

    int64_t now = TimeUtil::NowMicros();
    int64_t elapsed_us = now - it->second;
    int64_t min_us = config_.min_interval_between_same_type_ms * 1000;

    if (cmd.priority == PlcCommandPriority::CRITICAL) {
        return false;
    }

    return elapsed_us < min_us;
}

bool PlcCommandManager::SubmitCommand(const PlcCommand& cmd) {
    if (!running_.load()) return false;

    if (cmd.safety_level_value < config_.min_safety_level_for_command) {
        return false;
    }

    switch (cmd.type) {
        case PlcCommandType::EMERGENCY_STOP:
            if (!config_.enable_emergency_stop) return false;
            break;
        case PlcCommandType::SPEED_LIMIT:
        case PlcCommandType::TROLLEY_SLOW:
        case PlcCommandType::HOIST_HOLD:
            if (!config_.enable_speed_limit) return false;
            break;
        case PlcCommandType::SWING_DAMPING:
            if (!config_.enable_swing_damping) return false;
            break;
        default:
            break;
    }

    if (ShouldThrottle(cmd)) {
        stats_.commands_dropped++;
        return false;
    }

    PlcCommand cmd_with_id = cmd;
    cmd_with_id.command_id = next_command_id_++;
    cmd_with_id.timestamp_us = TimeUtil::NowMicros();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        if ((int)priority_queue_.size() >= config_.max_pending_commands) {
            if (cmd.priority <= PlcCommandPriority::NORMAL) {
                stats_.commands_dropped++;
                return false;
            }
        }

        priority_queue_.push(cmd_with_id);
        stats_.pending_count = (int)priority_queue_.size();
    }

    UpdateHistory(cmd_with_id);
    last_issue_time_us_[cmd.type] = cmd_with_id.timestamp_us;

    if (cmd.type == PlcCommandType::EMERGENCY_STOP) {
        emergency_active_.store(true);
        stats_.total_emergency_stops++;
        LOG_FATAL << "[PLC-ESTOP] Emergency STOP command issued! ID="
                  << cmd_with_id.command_id
                  << ", reason: " << cmd_with_id.reason;
    } else if (cmd.type == PlcCommandType::SPEED_LIMIT) {
        stats_.total_speed_limits++;
        LOG_WARN << "[PLC-SPEED] Speed limit command issued: "
                 << cmd_with_id.speed_limit_ratio * 100
                 << "% speed, track=" << cmd_with_id.track_id;
    } else if (cmd.type == PlcCommandType::SWING_DAMPING) {
        stats_.total_damping_commands++;
        LOG_WARN << "[PLC-DAMP] Swing damping command issued: gain="
                 << cmd_with_id.damping_gain
                 << ", track=" << cmd_with_id.track_id;
    }

    stats_.total_commands_issued++;
    UpdateSafetyLevel();

    if (command_callback_) {
        command_callback_(cmd_with_id);
    }

    return true;
}

PlcCommand PlcCommandManager::GetHighestPriorityCommand(int timeout_ms) {
    (void)timeout_ms;
    std::lock_guard<std::mutex> lock(queue_mutex_);

    if (priority_queue_.empty()) {
        return {};
    }

    PlcCommand cmd = priority_queue_.top();
    priority_queue_.pop();
    stats_.pending_count = (int)priority_queue_.size();
    return cmd;
}

bool PlcCommandManager::AckCommand(uint64_t command_id) {
    (void)command_id;
    stats_.commands_acked++;
    return true;
}

int PlcCommandManager::PendingCount() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return (int)priority_queue_.size();
}

void PlcCommandManager::UpdateHistory(const PlcCommand& cmd) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    command_history_.push_front(cmd);
    if ((int)command_history_.size() > config_.command_history_size) {
        command_history_.pop_back();
    }
}

void PlcCommandManager::UpdateSafetyLevel() {
    SafetyLevel max_level = SafetyLevel::SAFE;

    for (auto& entry : last_issue_time_us_) {
        int64_t now = TimeUtil::NowMicros();
        if (now - entry.second < 5 * 1000000) {
            if (entry.first == PlcCommandType::EMERGENCY_STOP) {
                max_level = SafetyLevel::CRITICAL;
                break;
            } else if (entry.first == PlcCommandType::SWING_DAMPING) {
                if (max_level < SafetyLevel::DANGER) max_level = SafetyLevel::DANGER;
            } else if (entry.first == PlcCommandType::SPEED_LIMIT) {
                if (max_level < SafetyLevel::WARNING) max_level = SafetyLevel::WARNING;
            }
        }
    }

    stats_.highest_active_level = max_level;
}

std::vector<PlcCommand> PlcCommandManager::GetRecentCommands(int max_count) const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    std::vector<PlcCommand> result;
    int count = std::min(max_count, (int)command_history_.size());
    result.reserve(count);
    for (int i = 0; i < count; i++) {
        result.push_back(command_history_[i]);
    }
    return result;
}

void PlcCommandManager::ClearEmergency() {
    emergency_active_.store(false);
    if (stats_.highest_active_level == SafetyLevel::CRITICAL) {
        stats_.highest_active_level = SafetyLevel::SAFE;
    }
    LOG_INFO << "[PLC-ESTOP] Emergency cleared";
}

}
