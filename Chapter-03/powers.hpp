#ifndef CHAPTER_03_POWERS_HPP
#define CHAPTER_03_POWERS_HPP

namespace ct {
namespace iterative {
// The iterative implementation, which has complexity O(n)

template<typename T>
T power(T x, unsigned n) {
    if (n == 0) {
        return T{1};
    }
    T result = x;

    while (--n > 0) {
        result *= x;
    }

    return result;
}
} // namespace iterative

namespace recursive {
// The recursive implementation, which has complexity O(log n)

template<typename T>
T power(T x, unsigned n) {
    if (n == 0) {
        return T{1};
    }
    if (n == 1) {
        return x;
    }
    const auto pow2 = power(x, n / 2);
    T result = (n % 2) == 0 ? T{1} : x;
    return result * pow2 * pow2;
}
} // namespace recursive
} // namespace ct

#endif // CHAPTER_03_POWERS_HPP