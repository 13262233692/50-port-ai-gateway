#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <deque>
#include <array>

namespace port_ai_gateway {

enum class PlcCommandType {
    NONE = 0,
    SPEED_LIMIT = 1,
    SWING_DAMPING = 2,
    EMERGENCY_STOP = 3,
    HOIST_HOLD = 4,
    TROLLEY_SLOW = 5,
    ACK = 6
};

enum class PlcCommandPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

enum class SafetyLevel {
    SAFE = 0,
    CAUTION = 1,
    WARNING = 2,
    DANGER = 3,
    CRITICAL = 4
};

struct PlcCommand {
    uint64_t command_id = 0;
    PlcCommandType type = PlcCommandType::NONE;
    PlcCommandPriority priority = PlcCommandPriority::NORMAL;

    int64_t timestamp_us = 0;
    int64_t deadline_us = 0;

    float speed_limit_ratio = 1.0f;
    float damping_gain = 0.0f;

    int track_id = -1;
    float swing_energy_joules = 0.0f;
    float safety_level_value = 0.0f;

    std::string reason;

    bool ack_required = false;
    bool is_cancel = false;

    bool operator<(const PlcCommand& other) const {
        if (priority != other.priority) {
            return static_cast<int>(priority) < static_cast<int>(other.priority);
        }
        return timestamp_us > other.timestamp_us;
    }
};

struct SwingCycleEnergy {
    int track_id = -1;
    int cycle_index = 0;
    double start_time_s = 0.0;
    double end_time_s = 0.0;
    double period_s = 0.0;

    float max_speed_m_s = 0.0f;
    float avg_speed_m_s = 0.0f;
    float amplitude_m = 0.0f;

    float kinetic_energy_joules = 0.0f;
    float peak_kinetic_energy = 0.0f;

    float yaw_angle_rad = 0.0f;
    float yaw_rate_rad_s = 0.0f;

    SafetyLevel safety_level = SafetyLevel::SAFE;
};

struct EnergyAnalyzerConfig {
    float spreader_mass_kg = 35000.0f;
    float gravity_m_s2 = 9.81f;
    float rope_length_m = 15.0f;

    float energy_warning_threshold_j = 5000.0f;
    float energy_danger_threshold_j = 15000.0f;
    float energy_critical_threshold_j = 30000.0f;

    float speed_warning_m_s = 1.5f;
    float speed_danger_m_s = 2.5f;
    float speed_critical_m_s = 4.0f;

    float amplitude_warning_m = 0.5f;
    float amplitude_danger_m = 1.2f;
    float amplitude_critical_m = 2.0f;

    float min_cycle_period_s = 2.0f;
    float max_cycle_period_s = 15.0f;

    int min_points_per_cycle = 15;

    float yaw_warning_rad = 0.2f;
    float yaw_danger_rad = 0.5f;

    int max_history_cycles = 50;
};

struct TrackEnergyStatus {
    int track_id = -1;
    SafetyLevel safety_level = SafetyLevel::SAFE;
    float current_energy_j = 0.0f;
    float current_speed_m_s = 0.0f;
    float current_amplitude_m = 0.0f;
    float current_yaw_rad = 0.0f;

    float swing_period_s = 0.0f;
    int completed_cycles = 0;

    std::deque<SwingCycleEnergy> cycle_history;
};

class SwingEnergyAnalyzer {
public:
    SwingEnergyAnalyzer();
    ~SwingEnergyAnalyzer();

    bool Init(const EnergyAnalyzerConfig& config);
    void Release();

    void UpdateTrack(int track_id, const Vec3& position, const Vec3& velocity,
                     int64_t timestamp_us);

    void RemoveTrack(int track_id);

    SafetyLevel GetTrackSafetyLevel(int track_id) const;
    float GetTrackEnergy(int track_id) const;
    float GetTrackSwingPeriod(int track_id) const;

    std::vector<TrackEnergyStatus> GetAllStatuses() const;

    bool ShouldTriggerDamping(int track_id, PlcCommand& out_cmd) const;

    const EnergyAnalyzerConfig& GetConfig() const { return config_; }

    void Reset();

private:
    EnergyAnalyzerConfig config_;

    struct TrackState {
        int track_id = -1;

        std::deque<Vec3> position_history;
        std::deque<Vec3> velocity_history;
        std::deque<int64_t> timestamp_history;

        std::deque<SwingCycleEnergy> cycle_history;
        int cycle_counter = 0;

        Vec3 last_zero_cross_pos = {0, 0, 0};
        int64_t last_zero_cross_time_us = 0;
        int zero_cross_count = 0;

        float cycle_integral_j = 0.0f;
        float cycle_peak_speed = 0.0f;
        float cycle_peak_energy = 0.0f;
        float cycle_max_amplitude = 0.0f;
        int64_t cycle_start_us = 0;

        float current_energy = 0.0f;
        float current_speed = 0.0f;
        float current_amplitude = 0.0f;
        float current_yaw = 0.0f;
        float swing_period_s = 0.0f;

        SafetyLevel safety_level = SafetyLevel::SAFE;

        Vec3 swing_center = {0, 0, 0};
        bool center_calibrated = false;
    };

    std::vector<std::shared_ptr<TrackState>> tracks_;

    std::shared_ptr<TrackState> GetOrCreateTrack(int track_id);
    std::shared_ptr<TrackState> FindTrack(int track_id) const;

    void UpdateSwingCenter(TrackState& state, const Vec3& pos);

    bool DetectZeroCrossing(TrackState& state, const Vec3& pos, const Vec3& vel,
                            int64_t timestamp_us);

    void IntegrateKineticEnergy(TrackState& state, const Vec3& velocity,
                                float dt_s);

    void ComputeSwingMetrics(TrackState& state, const Vec3& pos,
                             const Vec3& vel);

    void CompleteCycle(TrackState& state, int64_t end_time_us);

    SafetyLevel EvaluateSafetyLevel(const TrackState& state) const;

    float CalculateYawAngle(const TrackState& state) const;
};

}
