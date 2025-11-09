#ifndef CHAPTER_13_CACHING_FUNCTION_HPP
#define CHAPTER_13_CACHING_FUNCTION_HPP

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>


namespace ct {

template <typename F>
class CachingFunction {
    std::recursive_mutex lock_;
    std::unordered_map<std::string, std::unique_ptr<std::string>> cache_;
    F function_;

public:

    CachingFunction(F&& function) : function_(std::forward<F>(function)) {}

    const std::string& operator()(const std::string& arg) {
        std::lock_guard lock(lock_);

        auto& cached = cache_[arg];
        if (!cached) {
            cached = std::make_unique<std::string>(function_(arg));
        }

        return *cached;
    }
};


}

#endif //CHAPTER_13_CACHING_FUNCTION_HPP