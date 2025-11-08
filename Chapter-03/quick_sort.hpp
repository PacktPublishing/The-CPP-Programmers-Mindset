#ifndef CHAPTER_03_QUICK_SORT_HPP
#define CHAPTER_03_QUICK_SORT_HPP

#include <cstdint>
#include <span>
#include <utility>

namespace ct {
template<typename T>
std::ptrdiff_t partition(std::span<T> data) {
    auto &pivot_value = data.back();
    std::ptrdiff_t pivot_pos = 0;
    for (std::size_t j = 0; j < data.size() - 1; ++j) {
        if (data[j] <= pivot_value) {
            std::swap(data[pivot_pos], data[j]);
            ++pivot_pos;
        }
    }
    std::swap(data[pivot_pos], pivot_value);
    return pivot_pos;
}

template<typename T>
void quicksort(std::span<T> data) {
    if (data.size() < 2) {
        return;
    }
    auto pivot = partition(data);

    quicksort(data.subspan(0, pivot));
    quicksort(data.subspan(pivot + 1));
}
} // namespace ct

#endif // CHAPTER_03_QUICK_SORT_HPP