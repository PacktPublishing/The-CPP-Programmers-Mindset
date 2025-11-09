#ifndef CHAPTER_13_TASK_HPP
#define CHAPTER_13_TASK_HPP

#include <thread>
#include <mutex>
#include <condition_variable>

namespace ct {

class Task {
    std::mutex lock_;
    std::condition_variable condition_;
    std::thread worker_;
    bool is_finished_ = false;

public:

    template <typename F>
    Task(F&& func)
        : worker_([f=std::forward<F>(func), this]() {
            f();
            std::unique_lock lk(lock_);
            is_finished_ = true;
            lk.unlock();
            condition_.notify_all();
        })
    {}

    ~Task() { worker_.join(); }

    bool is_finished() {
        std::lock_guard lk(lock_);
        return is_finished_;
    }

    void wait() {
        std::unique_lock lk(lock_);
        condition_.wait(lk, [this] { return is_finished_; });
    }
};

} // namespace ct
#endif //CHAPTER_13_TASK_HPP
