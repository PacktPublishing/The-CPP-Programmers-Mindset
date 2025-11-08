#ifndef CHAPTER_02_CONVERSION_TRAITS_HPP
#define CHAPTER_02_CONVERSION_TRAITS_HPP

#include <stdexcept>
#include <concepts>
#include <limits>
#include <bits/error_constants.h>


namespace ct {
template<typename From, typename To, typename=void>
struct ExactConversionTraits {
    using from_ref = const From &;
    using to_ref = To &;

    static void convert(to_ref to, from_ref from) {
        throw std::runtime_error("invalid exact conversion");
    }
};


template<std::integral From, std::integral To>
struct ExactConversionTraits<From, To, void> {
    using from_ref = const From &;
    using to_ref = To &;

    static void convert(to_ref to, from_ref from) {
        if (from > std::numeric_limits < To > ::max() ||
            from < std::numeric_limits < To > ::lowest()) {
            throw std::runtime_error("invalid exact conversion");
        }
        to = static_cast<To>(from);
    }
};


template<>
struct ExactConversionTraits<int64_t, double> {
    using from_ref = const int64_t &;
    using to_ref = double &;

    static void convert(to_ref to, from_ref from) {
        static constexpr int64_t double_max_int = (1ULL << (1 + std::numeric_limits<double>::digits())) - 1ULL;
        if (from < -double_max_int || from > double_max_int) {
            throw std::runtime_error("invalid exact conversion");
        }
        to = static_cast<double>(from);
    }
};
} // namespace ct

#endif //CHAPTER_02_CONVERSION_TRAITS_HPP