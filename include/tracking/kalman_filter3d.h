#pragma once

#include <array>
#include <cstddef>
#include <cmath>

namespace port_ai_gateway {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
    Vec3 operator*(float s) const {
        return {x * s, y * s, z * s};
    }
    float Dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    float Norm() const {
        return std::sqrt(Dot(*this));
    }
    Vec3 Normalized() const {
        float n = Norm();
        if (n < 1e-8f) return {0, 0, 0};
        return {x / n, y / n, z / n};
    }
};

struct Vec6 {
    std::array<float, 6> data{};

    float& operator[](size_t i) { return data[i]; }
    float operator[](size_t i) const { return data[i]; }

    Vec6 operator+(const Vec6& other) const {
        Vec6 r;
        for (size_t i = 0; i < 6; i++) r[i] = data[i] + other[i];
        return r;
    }
    Vec6 operator-(const Vec6& other) const {
        Vec6 r;
        for (size_t i = 0; i < 6; i++) r[i] = data[i] - other[i];
        return r;
    }
};

struct Mat6 {
    std::array<std::array<float, 6>, 6> data{};

    std::array<float, 6>& operator[](size_t i) { return data[i]; }
    const std::array<float, 6>& operator[](size_t i) const { return data[i]; }

    static Mat6 Identity() {
        Mat6 m;
        for (size_t i = 0; i < 6; i++) m[i][i] = 1.0f;
        return m;
    }

    static Mat6 Zero() {
        Mat6 m;
        return m;
    }

    Mat6 Transpose() const {
        Mat6 r;
        for (size_t i = 0; i < 6; i++)
            for (size_t j = 0; j < 6; j++)
                r[j][i] = data[i][j];
        return r;
    }

    Mat6 operator+(const Mat6& other) const {
        Mat6 r;
        for (size_t i = 0; i < 6; i++)
            for (size_t j = 0; j < 6; j++)
                r[i][j] = data[i][j] + other[i][j];
        return r;
    }

    Mat6 operator-(const Mat6& other) const {
        Mat6 r;
        for (size_t i = 0; i < 6; i++)
            for (size_t j = 0; j < 6; j++)
                r[i][j] = data[i][j] - other[i][j];
        return r;
    }

    Mat6 operator*(const Mat6& other) const {
        Mat6 r;
        for (size_t i = 0; i < 6; i++) {
            for (size_t k = 0; k < 6; k++) {
                float a = data[i][k];
                if (std::abs(a) < 1e-12f) continue;
                for (size_t j = 0; j < 6; j++) {
                    r[i][j] += a * other[k][j];
                }
            }
        }
        return r;
    }

    Vec6 MulVec(const Vec6& v) const {
        Vec6 r;
        for (size_t i = 0; i < 6; i++) {
            float s = 0.0f;
            for (size_t j = 0; j < 6; j++) {
                s += data[i][j] * v[j];
            }
            r[i] = s;
        }
        return r;
    }
};

struct Kalman3DConfig {
    float process_noise_pos = 0.01f;
    float process_noise_vel = 0.5f;
    float measurement_noise = 0.1f;
    float initial_cov_pos = 1.0f;
    float initial_cov_vel = 10.0f;
    float dt_default = 0.033f;
};

class KalmanFilter3D {
public:
    KalmanFilter3D();

    void Init(const Vec3& initial_pos, const Kalman3DConfig& config);

    void Predict(float dt = -1.0f);

    void Update(const Vec3& measurement);

    Vec3 GetPosition() const { return {x_[0], x_[1], x_[2]}; }
    Vec3 GetVelocity() const { return {x_[3], x_[4], x_[5]}; }

    const Vec6& State() const { return x_; }
    const Mat6& Covariance() const { return P_; }

    float GetPositionUncertainty() const;

    bool IsInitialized() const { return initialized_; }

private:
    bool initialized_ = false;
    Kalman3DConfig config_;

    Vec6 x_;
    Mat6 P_;

    void BuildF(Mat6& F, float dt) const;
    void BuildH(Mat6& H) const;
    void BuildQ(Mat6& Q, float dt) const;
    void BuildR(Mat6& R) const;

    bool Invert6x6TopLeft(const Mat6& M, Mat6& out, int size = 6) const;
};

}
