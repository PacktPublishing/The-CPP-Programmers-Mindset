#include "intensity_grid_search.hpp"

#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>

ct::Pos ct::find_closest_signal_grid_search(
    int dim_x, int dim_y, double detection_intensity, Pos start) {
    std::vector<Pos> signals;
    signals.reserve(dim_x * dim_y);

    for (int x = 0; x < dim_x; ++x) {
        for (int y = 0; y < dim_y; ++y) {
            if (compute_signal_intensity(x, y) > detection_intensity) {
                signals.emplace_back(x, y);
            }
        }
    }

    auto dist_to_start = [&start](const Pos &pos) {
        return std::max(std::abs(pos.x - start.x), std::abs(pos.y - start.y));
    };

    auto ordering = [&dist_to_start](const Pos &p1, const Pos &p2) {
        return dist_to_start(p1) < dist_to_start(p2);
    };

    auto closest_pos = std::min_element(signals.begin(), signals.end(), ordering);

    // this wasn't in the code sample, but is needed to complete the code
    if (closest_pos == signals.end()) {
        throw std::runtime_error("No signal found");
    }

    return *closest_pos;
}