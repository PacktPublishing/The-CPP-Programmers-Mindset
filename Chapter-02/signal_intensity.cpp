#include "signal_intensity.hpp"


#include <array>
#include <cmath>

namespace {
struct SourcePos {
    int x;
    int y;
    double intensity;
    double decay;
};


constexpr std::array<SourcePos, 3> sources{
    SourcePos{4, 4, 7.0, 2.0},
    SourcePos{5, 0, 5.0, 4.0},
    SourcePos{7, 5, 4.0, 0.2}
};
} // namespace

double ct::compute_signal_intensity(int x, int y) {
    double intensity = 0.0;

    for (const auto &source: sources) {
        const auto dx = static_cast<double>(x - source.x);
        const auto dy = static_cast<double>(y - source.y);

        intensity += source.intensity * std::exp(-source.decay * (dx * dx + dy * dy));
    }

    return intensity;
}