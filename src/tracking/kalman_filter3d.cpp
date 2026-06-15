#include "tracking/kalman_filter3d.h"
#include "common/logger.h"

#include <cmath>
#include <algorithm>

namespace port_ai_gateway {

KalmanFilter3D::KalmanFilter3D() {
    x_ = {};
    P_ = Mat6::Identity();
}

void KalmanFilter3D::Init(const Vec3& initial_pos, const Kalman3DConfig& config) {
    config_ = config;

    x_[0] = initial_pos.x;
    x_[1] = initial_pos.y;
    x_[2] = initial_pos.z;
    x_[3] = 0.0f;
    x_[4] = 0.0f;
    x_[5] = 0.0f;

    P_ = Mat6::Zero();
    for (int i = 0; i < 3; i++) {
        P_[i][i] = config_.initial_cov_pos;
    }
    for (int i = 3; i < 6; i++) {
        P_[i][i] = config_.initial_cov_vel;
    }

    initialized_ = true;
}

void KalmanFilter3D::BuildF(Mat6& F, float dt) const {
    F = Mat6::Identity();
    for (int i = 0; i < 3; i++) {
        F[i][i + 3] = dt;
    }
}

void KalmanFilter3D::BuildH(Mat6& H) const {
    H = Mat6::Zero();
    for (int i = 0; i < 3; i++) {
        H[i][i] = 1.0f;
    }
}

void KalmanFilter3D::BuildQ(Mat6& Q, float dt) const {
    Q = Mat6::Zero();

    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;

    float sigma_a_pos = config_.process_noise_pos;
    float sigma_a_vel = config_.process_noise_vel;

    for (int i = 0; i < 3; i++) {
        Q[i][i] = 0.25f * dt4 * sigma_a_pos;
        Q[i][i + 3] = 0.5f * dt3 * sigma_a_pos;
        Q[i + 3][i] = 0.5f * dt3 * sigma_a_pos;
        Q[i + 3][i + 3] = dt2 * sigma_a_vel;
    }
}

void KalmanFilter3D::BuildR(Mat6& R) const {
    R = Mat6::Zero();
    for (int i = 0; i < 3; i++) {
        R[i][i] = config_.measurement_noise;
    }
}

void KalmanFilter3D::Predict(float dt) {
    if (!initialized_) return;

    float use_dt = (dt > 0) ? dt : config_.dt_default;

    Mat6 F;
    BuildF(F, use_dt);

    Mat6 Q;
    BuildQ(Q, use_dt);

    x_ = F.MulVec(x_);

    Mat6 F_trans = F.Transpose();
    P_ = F * P_ * F_trans + Q;
}

void KalmanFilter3D::Update(const Vec3& measurement) {
    if (!initialized_) return;

    Mat6 H;
    BuildH(H);

    Mat6 H_trans = H.Transpose();

    Mat6 R;
    BuildR(R);

    Vec6 z;
    z[0] = measurement.x;
    z[1] = measurement.y;
    z[2] = measurement.z;
    z[3] = 0.0f;
    z[4] = 0.0f;
    z[5] = 0.0f;

    Vec6 y = z - H.MulVec(x_);

    Mat6 S = H * P_ * H_trans + R;

    Mat6 S_inv;
    if (!Invert6x6TopLeft(S, S_inv, 3)) {
        LOG_WARN << "Kalman3D: S matrix singular, skip update";
        return;
    }

    Mat6 K = P_ * H_trans * S_inv;

    x_ = x_ + K.MulVec(y);

    Mat6 I = Mat6::Identity();
    P_ = (I - K * H) * P_;
}

float KalmanFilter3D::GetPositionUncertainty() const {
    float trace = 0.0f;
    for (int i = 0; i < 3; i++) {
        trace += P_[i][i];
    }
    return std::sqrt(std::max(0.0f, trace));
}

bool KalmanFilter3D::Invert6x6TopLeft(const Mat6& M, Mat6& out, int size) const {
    Mat6 aug;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            aug[i][j] = M[i][j];
        }
        for (int j = size; j < 2 * size; j++) {
            aug[i][j] = (i == (j - size)) ? 1.0f : 0.0f;
        }
    }

    const int n = size;
    const int m = 2 * size;

    for (int col = 0; col < n; col++) {
        int pivot_row = col;
        float max_val = std::abs(aug[col][col]);
        for (int row = col + 1; row < n; row++) {
            float v = std::abs(aug[row][col]);
            if (v > max_val) {
                max_val = v;
                pivot_row = row;
            }
        }

        if (max_val < 1e-12f) {
            return false;
        }

        if (pivot_row != col) {
            std::swap(aug[col], aug[pivot_row]);
        }

        float pivot = aug[col][col];
        for (int j = col; j < m; j++) {
            aug[col][j] /= pivot;
        }

        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            float factor = aug[row][col];
            if (std::abs(factor) < 1e-12f) continue;
            for (int j = col; j < m; j++) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    out = Mat6::Zero();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out[i][j] = aug[i][j + n];
        }
    }
    return true;
}

}
