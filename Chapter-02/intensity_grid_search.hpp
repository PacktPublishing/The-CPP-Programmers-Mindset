#ifndef CHAPTER_02_INTENSITY_GRID_SEARCH_HPP
#define CHAPTER_02_INTENSITY_GRID_SEARCH_HPP

#include "signal_intensity.hpp"

namespace ct {
Pos find_closest_signal_grid_search(
    int dim_x, int dim_y, double signal_intensity, Pos start
);
} // namespace ct

#endif //CHAPTER_02_INTENSITY_GRID_SEARCH_HPP