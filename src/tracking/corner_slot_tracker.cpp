#include "tracking/corner_slot_tracker.h"
#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <set>

namespace port_ai_gateway {

CornerSlotTrack::CornerSlotTrack(int track_id, const CornerSlot3D& detection,
                                 int64_t timestamp_us, const Kalman3DConfig& kf_config)
    : track_id_(track_id)
    , state_(TrackState::TENTATIVE)
    , hit_streak_(1)
    , age_(1)
    , time_since_update_(0) {

    Vec3 pos(detection.x, detection.y, detection.z);
    kf_.Init(pos, kf_config);

    TrackTrajectoryPoint p;
    p.timestamp_us = timestamp_us;
    p.position = pos;
    p.velocity = {0, 0, 0};
    p.confidence = detection.confidence;
    trajectory_.push_back(p);
}

void CornerSlotTrack::Predict(float dt) {
    kf_.Predict(dt);
    age_++;
    time_since_update_++;
}

void CornerSlotTrack::Update(const CornerSlot3D& detection, int64_t timestamp_us) {
    Vec3 meas(detection.x, detection.y, detection.z);
    kf_.Update(meas);

    hit_streak_++;
    time_since_update_ = 0;

    if (state_ == TrackState::TENTATIVE && hit_streak_ >= 3) {
        state_ = TrackState::CONFIRMED;
    }

    TrackTrajectoryPoint p;
    p.timestamp_us = timestamp_us;
    p.position = kf_.GetPosition();
    p.velocity = kf_.GetVelocity();
    p.confidence = detection.confidence;
    trajectory_.push_back(p);

    if ((int)trajectory_.size() > max_trajectory_len_) {
        trajectory_.pop_front();
    }
}

void CornerSlotTrack::MarkMissed() {
    hit_streak_ = 0;
    if (state_ == TrackState::TENTATIVE) {
        state_ = TrackState::DELETED;
    } else if (time_since_update_ > 30) {
        state_ = TrackState::LOST;
    }
}

float CornerSlotTrack::AverageSpeed() const {
    if (trajectory_.size() < 2) return 0.0f;

    float total_speed = 0.0f;
    int count = 0;
    for (size_t i = 1; i < trajectory_.size(); i++) {
        auto& p0 = trajectory_[i - 1];
        auto& p1 = trajectory_[i];
        float dt_us = float(p1.timestamp_us - p0.timestamp_us);
        if (dt_us < 1.0f) continue;
        float dt = dt_us / 1e6f;
        Vec3 dp = p1.position - p0.position;
        float speed = dp.Norm() / dt;
        total_speed += speed;
        count++;
    }
    return count > 0 ? total_speed / count : 0.0f;
}

float CornerSlotTrack::SwingAmplitude() const {
    if (trajectory_.size() < 3) return 0.0f;

    Vec3 min_pos = trajectory_.front().position;
    Vec3 max_pos = trajectory_.front().position;

    for (auto& p : trajectory_) {
        min_pos.x = std::min(min_pos.x, p.position.x);
        min_pos.y = std::min(min_pos.y, p.position.y);
        min_pos.z = std::min(min_pos.z, p.position.z);
        max_pos.x = std::max(max_pos.x, p.position.x);
        max_pos.y = std::max(max_pos.y, p.position.y);
        max_pos.z = std::max(max_pos.z, p.position.z);
    }

    Vec3 range = max_pos - min_pos;
    return range.Norm() * 0.5f;
}

// ============================================================
// CornerSlotTracker
// ============================================================

CornerSlotTracker::CornerSlotTracker()
    : next_track_id_(1)
    , last_timestamp_us_(0) {
}

CornerSlotTracker::~CornerSlotTracker() {
    Release();
}

bool CornerSlotTracker::Init(const TrackerConfig& config) {
    config_ = config;
    next_track_id_ = 1;
    last_timestamp_us_ = 0;
    tracks_.clear();
    stats_ = {};
    return true;
}

void CornerSlotTracker::Release() {
    tracks_.clear();
    stats_ = {};
}

void CornerSlotTracker::Reset() {
    tracks_.clear();
    next_track_id_ = 1;
    last_timestamp_us_ = 0;
    stats_ = {};
}

float CornerSlotTracker::ComputeDistance(const CornerSlotTrackPtr& track,
                                         const CornerSlot3D& det) const {
    Vec3 pos = track->Position();
    Vec3 det_pos(det.x, det.y, det.z);
    Vec3 diff = pos - det_pos;
    return diff.Norm();
}

void CornerSlotTracker::GreedyMatch(const std::vector<CornerSlotTrackPtr>& tracks,
                                    const std::vector<CornerSlot3D>& detections,
                                    std::vector<int>& matched_tracks,
                                    std::vector<int>& unmatched_tracks,
                                    std::vector<int>& unmatched_dets) const {
    size_t num_tracks = tracks.size();
    size_t num_dets = detections.size();

    matched_tracks.assign(num_tracks, -1);
    unmatched_tracks.clear();
    unmatched_dets.clear();

    if (num_tracks == 0 || num_dets == 0) {
        for (size_t i = 0; i < num_tracks; i++) unmatched_tracks.push_back((int)i);
        for (size_t j = 0; j < num_dets; j++) unmatched_dets.push_back((int)j);
        return;
    }

    struct Pair {
        int track_idx;
        int det_idx;
        float distance;
        bool operator<(const Pair& other) const {
            return distance < other.distance;
        }
    };

    std::vector<Pair> pairs;
    pairs.reserve(num_tracks * num_dets);

    for (size_t i = 0; i < num_tracks; i++) {
        for (size_t j = 0; j < num_dets; j++) {
            float dist = ComputeDistance(tracks[i], detections[j]);
            if (dist <= config_.max_matching_distance) {
                pairs.push_back({(int)i, (int)j, dist});
            }
        }
    }

    std::sort(pairs.begin(), pairs.end());

    std::vector<bool> track_used(num_tracks, false);
    std::vector<bool> det_used(num_dets, false);

    for (auto& p : pairs) {
        if (!track_used[p.track_idx] && !det_used[p.det_idx]) {
            matched_tracks[p.track_idx] = p.det_idx;
            track_used[p.track_idx] = true;
            det_used[p.det_idx] = true;
        }
    }

    for (size_t i = 0; i < num_tracks; i++) {
        if (!track_used[i]) unmatched_tracks.push_back((int)i);
    }
    for (size_t j = 0; j < num_dets; j++) {
        if (!det_used[j]) unmatched_dets.push_back((int)j);
    }
}

void CornerSlotTracker::InitiateTrack(const CornerSlot3D& detection, int64_t timestamp_us) {
    if ((int)tracks_.size() >= config_.max_tracks) {
        LOG_WARN << "Tracker: max tracks (" << config_.max_tracks << ") reached, skip new track";
        return;
    }

    auto track = std::make_shared<CornerSlotTrack>(
        next_track_id_++, detection, timestamp_us, config_.kf_config);
    tracks_.push_back(track);

    LOG_DEBUG << "Tracker: initiated new track id=" << track->Id()
              << ", pos=(" << detection.x << "," << detection.y << "," << detection.z << ")";
}

void CornerSlotTracker::Update(const std::vector<CornerSlot3D>& detections,
                               int64_t timestamp_us) {
    float dt = 0.033f;
    if (last_timestamp_us_ > 0 && timestamp_us > last_timestamp_us_) {
        dt = (timestamp_us - last_timestamp_us_) / 1e6f;
    }
    dt = std::min(dt, 1.0f);

    for (auto& track : tracks_) {
        track->Predict(dt);
    }

    std::vector<CornerSlot3D> filtered_dets;
    for (auto& d : detections) {
        if (d.confidence >= config_.min_detection_confidence) {
            filtered_dets.push_back(d);
        }
    }

    std::vector<int> matched_tracks;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_dets;
    GreedyMatch(tracks_, filtered_dets, matched_tracks, unmatched_tracks, unmatched_dets);

    for (size_t i = 0; i < tracks_.size(); i++) {
        if (matched_tracks[i] >= 0) {
            tracks_[i]->Update(filtered_dets[matched_tracks[i]], timestamp_us);
        } else {
            tracks_[i]->MarkMissed();
        }
    }

    for (int det_idx : unmatched_dets) {
        InitiateTrack(filtered_dets[det_idx], timestamp_us);
    }

    std::vector<CornerSlotTrackPtr> new_tracks;
    new_tracks.reserve(tracks_.size());
    int deleted = 0, lost = 0, confirmed = 0, tentative = 0;
    for (auto& track : tracks_) {
        if (track->State() == TrackState::DELETED) {
            deleted++;
            continue;
        }
        if (track->TimeSinceUpdate() > config_.max_age) {
            deleted++;
            continue;
        }
        new_tracks.push_back(track);
        switch (track->State()) {
            case TrackState::TENTATIVE: tentative++; break;
            case TrackState::CONFIRMED: confirmed++; break;
            case TrackState::LOST: lost++; break;
            default: break;
        }
    }
    tracks_ = std::move(new_tracks);

    stats_.total_tracks = (int)tracks_.size();
    stats_.active_tracks = confirmed;
    stats_.tentative_tracks = tentative;
    stats_.lost_tracks = lost;
    stats_.last_timestamp_us = timestamp_us;

    last_timestamp_us_ = timestamp_us;
}

std::vector<CornerSlotTrackPtr> CornerSlotTracker::GetActiveTracks() const {
    std::vector<CornerSlotTrackPtr> result;
    for (auto& t : tracks_) {
        if (t->State() == TrackState::CONFIRMED) {
            result.push_back(t);
        }
    }
    return result;
}

std::vector<CornerSlotTrackPtr> CornerSlotTracker::GetConfirmedTracks() const {
    return GetActiveTracks();
}

CornerSlotTrackPtr CornerSlotTracker::GetTrackById(int track_id) const {
    for (auto& t : tracks_) {
        if (t->Id() == track_id) return t;
    }
    return nullptr;
}

}
