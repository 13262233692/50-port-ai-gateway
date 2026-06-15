#include "safety/swing_energy_analyzer.h"
#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace port_ai_gateway {

SwingEnergyAnalyzer::SwingEnergyAnalyzer() = default;
SwingEnergyAnalyzer::~SwingEnergyAnalyzer() { Release(); }

bool SwingEnergyAnalyzer::Init(const EnergyAnalyzerConfig& config) {
    config_ = config;
    tracks_.clear();
    LOG_INFO << "SwingEnergyAnalyzer initialized: mass=" << config_.spreader_mass_kg
             << "kg, rope=" << config_.rope_length_m << "m, "
             << "warn=" << config_.energy_warning_threshold_j
             << "J, danger=" << config_.energy_danger_threshold_j
             << "J, critical=" << config_.energy_critical_threshold_j << "J";
    return true;
}

void SwingEnergyAnalyzer::Release() {
    tracks_.clear();
}

void SwingEnergyAnalyzer::Reset() {
    tracks_.clear();
}

std::shared_ptr<SwingEnergyAnalyzer::TrackState>
SwingEnergyAnalyzer::FindTrack(int track_id) const {
    for (auto& t : tracks_) {
        if (t->track_id == track_id) return t;
    }
    return nullptr;
}

std::shared_ptr<SwingEnergyAnalyzer::TrackState>
SwingEnergyAnalyzer::GetOrCreateTrack(int track_id) {
    auto existing = FindTrack(track_id);
    if (existing) return existing;

    auto state = std::make_shared<TrackState>();
    state->track_id = track_id;
    state->cycle_start_us = 0;
    state->cycle_integral_j = 0.0f;
    state->cycle_peak_speed = 0.0f;
    state->cycle_peak_energy = 0.0f;
    state->cycle_max_amplitude = 0.0f;
    state->zero_cross_count = 0;
    state->cycle_counter = 0;
    state->center_calibrated = false;
    state->swing_center = {0, 0, 0};
    state->safety_level = SafetyLevel::SAFE;
    tracks_.push_back(state);
    return state;
}

void SwingEnergyAnalyzer::RemoveTrack(int track_id) {
    auto it = std::remove_if(tracks_.begin(), tracks_.end(),
        [track_id](const std::shared_ptr<TrackState>& t) {
            return t->track_id == track_id;
        });
    if (it != tracks_.end()) {
        LOG_DEBUG << "EnergyAnalyzer: removed track " << track_id;
        tracks_.erase(it, tracks_.end());
    }
}

void SwingEnergyAnalyzer::UpdateSwingCenter(TrackState& state, const Vec3& pos) {
    const int max_calib = 60;

    if (!state.center_calibrated) {
        state.position_history.push_back(pos);
        if ((int)state.position_history.size() > max_calib) {
            state.position_history.pop_front();
        }

        if ((int)state.position_history.size() >= 30) {
            Vec3 sum{0, 0, 0};
            for (auto& p : state.position_history) {
                sum = sum + p;
            }
            float n = (float)state.position_history.size();
            state.swing_center = {sum.x / n, sum.y / n, sum.z / n};
            state.center_calibrated = true;
            LOG_INFO << "EnergyAnalyzer: track " << state.track_id
                     << " swing center calibrated at ("
                     << state.swing_center.x << ", "
                     << state.swing_center.y << ", "
                     << state.swing_center.z << ")";
        }
    } else {
        state.position_history.push_back(pos);
        if ((int)state.position_history.size() > max_calib) {
            state.position_history.pop_front();
        }

        Vec3 sum{0, 0, 0};
        for (auto& p : state.position_history) {
            sum = sum + p;
        }
        float n = (float)state.position_history.size();
        Vec3 new_center{sum.x / n, sum.y / n, sum.z / n};

        float alpha = 0.005f;
        state.swing_center.x = state.swing_center.x * (1 - alpha) + new_center.x * alpha;
        state.swing_center.y = state.swing_center.y * (1 - alpha) + new_center.y * alpha;
        state.swing_center.z = state.swing_center.z * (1 - alpha) + new_center.z * alpha;
    }
}

bool SwingEnergyAnalyzer::DetectZeroCrossing(TrackState& state, const Vec3& pos,
                                             const Vec3& vel, int64_t timestamp_us) {
    if (!state.center_calibrated) return false;

    Vec3 rel_pos = pos - state.swing_center;

    float radial_speed = rel_pos.Dot(vel);
    if (rel_pos.Norm() < 1e-6f) return false;
    radial_speed /= rel_pos.Norm();

    bool crossed = false;

    if (state.last_zero_cross_time_us == 0) {
        state.last_zero_cross_time_us = timestamp_us;
        state.last_zero_cross_pos = pos;
        state.cycle_start_us = timestamp_us;
        state.cycle_integral_j = 0.0f;
        state.cycle_peak_speed = 0.0f;
        state.cycle_peak_energy = 0.0f;
        state.cycle_max_amplitude = 0.0f;
        return false;
    }

    float prev_radial_speed = 0.0f;
    if (state.velocity_history.size() >= 2 && state.position_history.size() >= 2) {
        auto prev_pos = state.position_history[state.position_history.size() - 2];
        auto prev_vel = state.velocity_history[state.velocity_history.size() - 2];
        Vec3 prev_rel = prev_pos - state.swing_center;
        if (prev_rel.Norm() > 1e-6f) {
            prev_radial_speed = prev_rel.Dot(prev_vel) / prev_rel.Norm();
        }
    }

    if (prev_radial_speed < 0 && radial_speed >= 0) {
        crossed = true;
        state.zero_cross_count++;

        float dt_s = (timestamp_us - state.last_zero_cross_time_us) / 1e6f;

        if (dt_s >= config_.min_cycle_period_s * 0.8f &&
            dt_s <= config_.max_cycle_period_s * 2.0f) {
            CompleteCycle(state, timestamp_us);
        } else if (dt_s < config_.min_cycle_period_s * 0.8f) {
            LOG_DEBUG << "Track " << state.track_id << ": zero-cross too fast ("
                      << dt_s << "s), noise?";
            crossed = false;
        }

        state.last_zero_cross_time_us = timestamp_us;
        state.last_zero_cross_pos = pos;
    }

    return crossed;
}

void SwingEnergyAnalyzer::IntegrateKineticEnergy(TrackState& state,
                                                 const Vec3& velocity,
                                                 float dt_s) {
    float speed = velocity.Norm();
    float ke = 0.5f * config_.spreader_mass_kg * speed * speed;

    state.current_energy = ke;
    state.current_speed = speed;

    state.cycle_integral_j += ke * dt_s;
    state.cycle_peak_speed = std::max(state.cycle_peak_speed, speed);
    state.cycle_peak_energy = std::max(state.cycle_peak_energy, ke);
}

void SwingEnergyAnalyzer::ComputeSwingMetrics(TrackState& state, const Vec3& pos,
                                              const Vec3&) {
    if (!state.center_calibrated) return;

    Vec3 rel_pos = pos - state.swing_center;
    float amplitude = rel_pos.Norm();
    state.current_amplitude = amplitude;

    state.cycle_max_amplitude = std::max(state.cycle_max_amplitude, amplitude);
}

float SwingEnergyAnalyzer::CalculateYawAngle(const TrackState& state) const {
    if (!state.center_calibrated || state.position_history.size() < 10) {
        return 0.0f;
    }

    auto& hist = state.position_history;
    size_t n = hist.size();

    Vec3 disp_sum{0, 0, 0};
    for (size_t i = n / 2; i < n; i++) {
        Vec3 rel = hist[i] - state.swing_center;
        disp_sum = disp_sum + Vec3(rel.x, 0, rel.z);
    }
    float count = (float)(n - n / 2);
    Vec3 avg_disp{disp_sum.x / count, 0, disp_sum.z / count};

    if (avg_disp.Norm() < 0.01f) return 0.0f;

    float yaw = std::atan2(avg_disp.x, avg_disp.z);
    return yaw;
}

void SwingEnergyAnalyzer::CompleteCycle(TrackState& state, int64_t end_time_us) {
    if (state.cycle_start_us == 0 || state.cycle_integral_j < 1e-6f) {
        state.cycle_start_us = end_time_us;
        state.cycle_integral_j = 0.0f;
        state.cycle_peak_speed = 0.0f;
        state.cycle_peak_energy = 0.0f;
        state.cycle_max_amplitude = 0.0f;
        return;
    }

    float period_s = (end_time_us - state.cycle_start_us) / 1e6f;
    if (period_s < config_.min_cycle_period_s * 0.5f) {
        return;
    }

    state.swing_period_s = period_s;
    state.cycle_counter++;

    float avg_speed = state.cycle_peak_speed * 0.637f;

    SwingCycleEnergy cycle;
    cycle.track_id = state.track_id;
    cycle.cycle_index = state.cycle_counter;
    cycle.start_time_s = state.cycle_start_us / 1e6;
    cycle.end_time_s = end_time_us / 1e6;
    cycle.period_s = period_s;
    cycle.max_speed_m_s = state.cycle_peak_speed;
    cycle.avg_speed_m_s = avg_speed;
    cycle.amplitude_m = state.cycle_max_amplitude;
    cycle.kinetic_energy_joules = state.cycle_integral_j;
    cycle.peak_kinetic_energy = state.cycle_peak_energy;
    cycle.yaw_angle_rad = CalculateYawAngle(state);
    cycle.yaw_rate_rad_s = cycle.yaw_angle_rad / period_s;

    if (cycle.kinetic_energy_joules >= config_.energy_critical_threshold_j ||
        cycle.max_speed_m_s >= config_.speed_critical_m_s ||
        cycle.amplitude_m >= config_.amplitude_critical_m) {
        cycle.safety_level = SafetyLevel::CRITICAL;
    } else if (cycle.kinetic_energy_joules >= config_.energy_danger_threshold_j ||
               cycle.max_speed_m_s >= config_.speed_danger_m_s ||
               cycle.amplitude_m >= config_.amplitude_danger_m) {
        cycle.safety_level = SafetyLevel::DANGER;
    } else if (cycle.kinetic_energy_joules >= config_.energy_warning_threshold_j ||
               cycle.max_speed_m_s >= config_.speed_warning_m_s ||
               cycle.amplitude_m >= config_.amplitude_warning_m) {
        cycle.safety_level = SafetyLevel::WARNING;
    } else {
        cycle.safety_level = SafetyLevel::SAFE;
    }

    state.cycle_history.push_back(cycle);
    if ((int)state.cycle_history.size() > config_.max_history_cycles) {
        state.cycle_history.pop_front();
    }

    state.safety_level = cycle.safety_level;

    const char* level_names[] = {"SAFE", "CAUTION", "WARNING", "DANGER", "CRITICAL"};
    int level_idx = static_cast<int>(cycle.safety_level);
    if (level_idx >= 2) {
        LOG_WARN << "[SWING-ENERGY] Track " << state.track_id
                 << " Cycle #" << state.cycle_counter
                 << " Level=" << level_names[level_idx]
                 << " Period=" << period_s << "s"
                 << " Amp=" << cycle.amplitude_m << "m"
                 << " Vmax=" << cycle.max_speed_m_s << "m/s"
                 << " E_cycle=" << cycle.kinetic_energy_joules << "J"
                 << " E_peak=" << cycle.peak_kinetic_energy << "J"
                 << " Yaw=" << cycle.yaw_angle_rad * 57.3f << "deg";
    }

    state.cycle_start_us = end_time_us;
    state.cycle_integral_j = 0.0f;
    state.cycle_peak_speed = 0.0f;
    state.cycle_peak_energy = 0.0f;
    state.cycle_max_amplitude = 0.0f;
}

SafetyLevel SwingEnergyAnalyzer::EvaluateSafetyLevel(const TrackState& state) const {
    SafetyLevel level = SafetyLevel::SAFE;

    if (state.current_energy >= config_.energy_critical_threshold_j ||
        state.current_speed >= config_.speed_critical_m_s ||
        state.current_amplitude >= config_.amplitude_critical_m) {
        level = SafetyLevel::CRITICAL;
    } else if (state.current_energy >= config_.energy_danger_threshold_j ||
               state.current_speed >= config_.speed_danger_m_s ||
               state.current_amplitude >= config_.amplitude_danger_m) {
        level = SafetyLevel::DANGER;
    } else if (state.current_energy >= config_.energy_warning_threshold_j ||
               state.current_speed >= config_.speed_warning_m_s ||
               state.current_amplitude >= config_.amplitude_warning_m) {
        level = SafetyLevel::WARNING;
    }

    return level;
}

void SwingEnergyAnalyzer::UpdateTrack(int track_id, const Vec3& position,
                                      const Vec3& velocity, int64_t timestamp_us) {
    auto state = GetOrCreateTrack(track_id);

    state->position_history.push_back(position);
    state->velocity_history.push_back(velocity);
    state->timestamp_history.push_back(timestamp_us);

    const int max_hist = 200;
    if ((int)state->position_history.size() > max_hist) {
        state->position_history.pop_front();
        state->velocity_history.pop_front();
        state->timestamp_history.pop_front();
    }

    UpdateSwingCenter(*state, position);

    float dt_s = 0.033f;
    if (state->timestamp_history.size() >= 2) {
        auto it = state->timestamp_history.end();
        --it;
        int64_t t2 = *it;
        --it;
        int64_t t1 = *it;
        if (t2 > t1) dt_s = (t2 - t1) / 1e6f;
    }
    dt_s = std::min(dt_s, 0.5f);

    IntegrateKineticEnergy(*state, velocity, dt_s);
    ComputeSwingMetrics(*state, position, velocity);

    DetectZeroCrossing(*state, position, velocity, timestamp_us);

    state->safety_level = EvaluateSafetyLevel(*state);

    state->current_yaw = CalculateYawAngle(*state);
}

SafetyLevel SwingEnergyAnalyzer::GetTrackSafetyLevel(int track_id) const {
    auto t = FindTrack(track_id);
    return t ? t->safety_level : SafetyLevel::SAFE;
}

float SwingEnergyAnalyzer::GetTrackEnergy(int track_id) const {
    auto t = FindTrack(track_id);
    return t ? t->current_energy : 0.0f;
}

float SwingEnergyAnalyzer::GetTrackSwingPeriod(int track_id) const {
    auto t = FindTrack(track_id);
    return t ? t->swing_period_s : 0.0f;
}

std::vector<TrackEnergyStatus> SwingEnergyAnalyzer::GetAllStatuses() const {
    std::vector<TrackEnergyStatus> result;
    result.reserve(tracks_.size());

    for (auto& t : tracks_) {
        TrackEnergyStatus s;
        s.track_id = t->track_id;
        s.safety_level = t->safety_level;
        s.current_energy_j = t->current_energy;
        s.current_speed_m_s = t->current_speed;
        s.current_amplitude_m = t->current_amplitude;
        s.current_yaw_rad = t->current_yaw;
        s.swing_period_s = t->swing_period_s;
        s.completed_cycles = t->cycle_counter;
        s.cycle_history = t->cycle_history;
        result.push_back(s);
    }
    return result;
}

bool SwingEnergyAnalyzer::ShouldTriggerDamping(int track_id, PlcCommand& out_cmd) const {
    auto t = FindTrack(track_id);
    if (!t) return false;

    SafetyLevel level = t->safety_level;

    if (level == SafetyLevel::SAFE || level == SafetyLevel::CAUTION) {
        return false;
    }

    out_cmd.track_id = track_id;
    out_cmd.swing_energy_joules = t->current_energy;
    out_cmd.safety_level_value = static_cast<float>(level);

    switch (level) {
        case SafetyLevel::WARNING:
            out_cmd.type = PlcCommandType::SPEED_LIMIT;
            out_cmd.priority = PlcCommandPriority::NORMAL;
            out_cmd.speed_limit_ratio = 0.7f;
            out_cmd.reason = "Swing energy exceeded warning threshold";
            break;
        case SafetyLevel::DANGER:
            out_cmd.type = PlcCommandType::SWING_DAMPING;
            out_cmd.priority = PlcCommandPriority::HIGH;
            out_cmd.speed_limit_ratio = 0.4f;
            out_cmd.damping_gain = 0.6f;
            out_cmd.reason = "Swing energy reached danger level";
            break;
        case SafetyLevel::CRITICAL:
            out_cmd.type = PlcCommandType::EMERGENCY_STOP;
            out_cmd.priority = PlcCommandPriority::CRITICAL;
            out_cmd.speed_limit_ratio = 0.0f;
            out_cmd.damping_gain = 1.0f;
            out_cmd.reason = "CRITICAL: Swing energy exceeded safety limit, "
                             "rope failure IMMINENT";
            out_cmd.ack_required = true;
            break;
        default:
            return false;
    }

    return true;
}

}
