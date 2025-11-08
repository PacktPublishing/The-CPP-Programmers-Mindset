#ifndef CHAPTER_02_EXPANDING_SEARCH_RANGE_HPP
#define CHAPTER_02_EXPANDING_SEARCH_RANGE_HPP


#include "signal_intensity.hpp"


namespace ct {
Pos find_closest_signal_range(
    int dim_x, int dim_y, double detection_intensity, Pos start
);
}

#endif //CHAPTER_02_EXPANDING_SEARCH_RANGE_HPP