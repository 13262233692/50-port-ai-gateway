#pragma once

#include <queue>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <functional>
#include <cstdint>
#include <memory>
#include <map>

#include "common/thread_safe_queue.h"
#include "safety/swing_energy_analyzer.h"

namespace port_ai_gateway {

using PlcCommandCallback = std::function<void(const PlcCommand&)>;

struct PlcManagerConfig {
    int max_pending_commands = 100;
    int command_history_size = 500;

    int min_interval_between_same_type_ms = 500;

    bool enable_emergency_stop = true;
    bool enable_speed_limit = true;
    bool enable_swing_damping = true;

    float min_safety_level_for_command = 2.0f;
};

struct PlcManagerStats {
    uint64_t total_commands_issued = 0;
    uint64_t total_emergency_stops = 0;
    uint64_t total_speed_limits = 0;
    uint64_t total_damping_commands = 0;
    uint64_t commands_dropped = 0;
    uint64_t commands_acked = 0;
    int pending_count = 0;

    SafetyLevel highest_active_level = SafetyLevel::SAFE;
};

class PlcCommandManager {
public:
    PlcCommandManager();
    ~PlcCommandManager();

    bool Init(const PlcManagerConfig& config);
    void Release();

    bool Start();
    void Stop();

    void SetCommandCallback(PlcCommandCallback cb) {
        command_callback_ = std::move(cb);
    }

    bool SubmitCommand(const PlcCommand& cmd);

    bool AckCommand(uint64_t command_id);

    PlcCommand GetHighestPriorityCommand(int timeout_ms = 100);

    int PendingCount() const;

    const PlcManagerStats& GetStats() const { return stats_; }

    SafetyLevel GetActiveSafetyLevel() const { return stats_.highest_active_level; }

    std::vector<PlcCommand> GetRecentCommands(int max_count = 20) const;

    bool IsEmergencyActive() const { return emergency_active_.load(); }
    void ClearEmergency();

private:
    PlcManagerConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> emergency_active_{false};

    struct PqComparator {
        bool operator()(const PlcCommand& a, const PlcCommand& b) const {
            return a < b;
        }
    };

    std::priority_queue<PlcCommand, std::vector<PlcCommand>, PqComparator> priority_queue_;
    mutable std::mutex queue_mutex_;

    std::deque<PlcCommand> command_history_;
    mutable std::mutex history_mutex_;

    std::map<PlcCommandType, int64_t> last_issue_time_us_;

    PlcCommandCallback command_callback_;
    PlcManagerStats stats_;

    uint64_t next_command_id_ = 1;

    bool ShouldThrottle(const PlcCommand& cmd) const;
    void UpdateHistory(const PlcCommand& cmd);
    void UpdateSafetyLevel();
};

}
