#ifndef CHAPTER_02_FUNCTION_WITH_NOISE_HPP
#define CHAPTER_02_FUNCTION_WITH_NOISE_HPP

#include <random>


namespace ct {
class FunctionWithNoise {
    std::mt19937 m_rng;
    std::normal_distribution<double> m_dist;

public:
    FunctionWithNoise() noexcept
        : m_rng{std::random_device{}()}, m_dist{0., 1.} {
    }

    FunctionWithNoise(size_t seed) noexcept
        : m_rng{seed}, m_dist{0., 1.} {
    }

    FunctionWithNoise(double mean, double stddev) noexcept
        : m_rng{std::random_device{}()}, m_dist{mean, stddev} {
    }

    FunctionWithNoise(size_t seed, double mean, double stddev) noexcept
        : m_rng{seed}, m_dist{mean, stddev} {
    }

    double operator()(double arg) noexcept {
        auto noise = m_dist(m_rng);
        return 2. * arg + 1 + noise;
    }
};
}

#endif //CHAPTER_02_FUNCTION_WITH_NOISE_HPP