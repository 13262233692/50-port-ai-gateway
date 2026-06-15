#pragma once

#include <vector>
#include <memory>
#include <deque>
#include <cstdint>
#include <cmath>

#include "common/frame.h"
#include "inference/corner_slot_detector.h"
#include "tracking/kalman_filter3d.h"

namespace port_ai_gateway {

enum class TrackState {
    TENTATIVE = 0,
    CONFIRMED = 1,
    LOST = 2,
    DELETED = 3
};

struct TrackTrajectoryPoint {
    int64_t timestamp_us = 0;
    Vec3 position;
    Vec3 velocity;
    float confidence = 0.0f;
};

class CornerSlotTrack {
public:
    CornerSlotTrack(int track_id, const CornerSlot3D& detection, int64_t timestamp_us,
                    const Kalman3DConfig& kf_config);

    void Predict(float dt = -1.0f);
    void Update(const CornerSlot3D& detection, int64_t timestamp_us);
    void MarkMissed();

    int Id() const { return track_id_; }
    TrackState State() const { return state_; }
    int HitStreak() const { return hit_streak_; }
    int Age() const { return age_; }
    int TimeSinceUpdate() const { return time_since_update_; }

    Vec3 Position() const { return kf_.GetPosition(); }
    Vec3 Velocity() const { return kf_.GetVelocity(); }

    float PositionUncertainty() const { return kf_.GetPositionUncertainty(); }

    const std::deque<TrackTrajectoryPoint>& Trajectory() const { return trajectory_; }

    float AverageSpeed() const;
    float SwingAmplitude() const;

    void SetState(TrackState s) { state_ = s; }

private:
    int track_id_ = -1;
    TrackState state_ = TrackState::TENTATIVE;
    int hit_streak_ = 1;
    int age_ = 1;
    int time_since_update_ = 0;
    int max_trajectory_len_ = 500;

    KalmanFilter3D kf_;
    std::deque<TrackTrajectoryPoint> trajectory_;
};

using CornerSlotTrackPtr = std::shared_ptr<CornerSlotTrack>;

struct TrackerConfig {
    Kalman3DConfig kf_config;

    int max_age = 60;
    int n_init = 3;
    float max_matching_distance = 2.0f;
    float min_detection_confidence = 0.3f;

    int max_trajectory_points = 500;

    int max_tracks = 16;
};

struct TrackerStats {
    int total_tracks = 0;
    int active_tracks = 0;
    int tentative_tracks = 0;
    int lost_tracks = 0;
    int64_t last_timestamp_us = 0;
};

class CornerSlotTracker {
public:
    CornerSlotTracker();
    ~CornerSlotTracker();

    bool Init(const TrackerConfig& config);
    void Release();

    void Update(const std::vector<CornerSlot3D>& detections, int64_t timestamp_us);

    std::vector<CornerSlotTrackPtr> GetActiveTracks() const;
    std::vector<CornerSlotTrackPtr> GetConfirmedTracks() const;

    CornerSlotTrackPtr GetTrackById(int track_id) const;

    const TrackerStats& GetStats() const { return stats_; }

    void Reset();

private:
    TrackerConfig config_;
    std::vector<CornerSlotTrackPtr> tracks_;
    int next_track_id_ = 1;
    int64_t last_timestamp_us_ = 0;
    TrackerStats stats_;

    float ComputeDistance(const CornerSlotTrackPtr& track, const CornerSlot3D& det) const;

    void GreedyMatch(const std::vector<CornerSlotTrackPtr>& tracks,
                     const std::vector<CornerSlot3D>& detections,
                     std::vector<int>& track_indices,
                     std::vector<int>& unmatched_tracks,
                     std::vector<int>& unmatched_dets) const;

    void InitiateTrack(const CornerSlot3D& detection, int64_t timestamp_us);
};

}
