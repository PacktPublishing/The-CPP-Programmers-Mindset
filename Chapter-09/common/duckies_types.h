#ifndef DUCKIES_COMMON_DUCKIES_TYPES_H
#define DUCKIES_COMMON_DUCKIES_TYPES_H

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace duckies {
struct Coordinate {
    float latitude = 0.0f;
    float longitude = 0.0f;
};

class RubberDuckData {
    std::vector<Coordinate> coordinates_;
    std::vector<std::chrono::year_month_day> dates_;
    std::vector<std::string> descriptions_;

public:
    const std::vector<Coordinate> &coordinates() const noexcept {
        return coordinates_;
    }

    const std::vector<std::chrono::year_month_day> dates() const noexcept {
        return dates_;
    }

    const std::vector<std::string> descriptions() const noexcept {
        return descriptions_;
    }

    void reserve_additional(size_t new_elements) {
        auto current_size = coordinates_.size();
        auto new_size = current_size + new_elements;

        coordinates_.reserve(new_size);
        dates_.reserve(new_size);
        descriptions_.reserve(new_size);
    }

    void insert(float latitude, float longitude,
                std::chrono::year_month_day date, std::string descr) {
        coordinates_.emplace_back(latitude, longitude);
        dates_.emplace_back(date);
        descriptions_.emplace_back(std::move(descr));
    }
};
}

#endif // DUCKIES_COMMON_DUCKIES_TYPES_H