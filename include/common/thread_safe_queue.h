#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>
#include <chrono>
#include <stdexcept>

namespace port_ai_gateway {

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 0)
        : max_size_(max_size)
        , running_(true) {}

    ~ThreadSafeQueue() {
        Stop();
    }

    void Push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size_ > 0) {
            not_full_.wait(lock, [this]() {
                return !running_ || queue_.size() < max_size_;
            });
        }
        if (!running_) {
            throw std::runtime_error("Queue is stopped");
        }
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    bool TryPush(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ || (max_size_ > 0 && queue_.size() >= max_size_)) {
            return false;
        }
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    T Pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() {
            return !running_ || !queue_.empty();
        });
        if (!running_ && queue_.empty()) {
            throw std::runtime_error("Queue is stopped and empty");
        }
        T item = std::move(queue_.front());
        queue_.pop();
        if (max_size_ > 0) {
            not_full_.notify_one();
        }
        return item;
    }

    bool TryPop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        if (max_size_ > 0) {
            not_full_.notify_one();
        }
        return true;
    }

    bool WaitPop(T& item, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
            [this]() { return !running_ || !queue_.empty(); })) {
            return false;
        }
        if (!running_ && queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        if (max_size_ > 0) {
            not_full_.notify_one();
        }
        return true;
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool Empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void Stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
        if (max_size_ > 0) {
            not_full_.notify_all();
        }
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t max_size_;
    std::atomic<bool> running_;
};

}
