#ifndef CHAPTER_02_FIND_ROOT_HPP
#define CHAPTER_02_FIND_ROOT_HPP

#include <cmath>
#include <limits>
#include <functional>

namespace ct {
/*
 * These helper functions are here to facilitate the bisection algorithm.
 * The first compares two floating numbers and returns true if either the
 * absolute difference or relative difference is less than epsilon (machine
 * precision).
 *
 * The second just computes the midpoint of two numbers.
 */


template<typename Real>
bool compare_reals_equal(Real a, Real b) {
    // Machine precision for the Real type
    constexpr Real tol = std::numeric_limits<Real>::epsilon();

    const auto abs_diff = std::abs(a - b);
    if (abs_diff <= tol) {
        return true;
    }

    const auto abs_a = std::abs(a);
    const auto abs_b = std::abs(b);
    const auto max_abs = std::max(abs_a, abs_b);
    return abs_diff <= tol * max_abs;
}


template<typename Real>
constexpr Real midpoint(Real a, Real b) {
    return (a + b) / Real{2};
}

template<typename Function, typename Real>
Real find_root_bisect(Function &&function, Real pos, Real neg, Real tol) {
    auto fpos = function(pos);

    // Driving loop
    while (!compare_reals_equal(pos, neg)) {
        auto m = midpoint(pos, neg);

        auto fm = function(m);

        // Quit early if the function is already (almost) 0.
        if (std::abs(fm) < tol) { return m; }

        // The decision logic to find the next point to check
        if (std::signbit(fm) == std::signbit(fpos)) {
            pos = m;
            fpos = fm;
        } else {
            neg = m;
        }
    }

    return pos;
}
} // namespace ct

#endif //CHAPTER_02_FIND_ROOT_HPP