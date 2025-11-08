#include "signal_intensity.hpp"

#include <gtest/gtest.h>


#include "intensity_grid_search.hpp"
#include "expanding_search_range.hpp"


TEST(SignalIntensity, TestGridSearchClosestSignal) {
    auto pos = ct::find_closest_signal_grid_search(6, 6, 5.0, {3, 3});

    // The closest signal is at (4, 4)
    EXPECT_EQ(pos.x, 4);
    EXPECT_EQ(pos.y, 4);
}


TEST(SignalIntenisty, TestGridSearchExpandingSearch) {
    auto pos = ct::find_closest_signal_range(6, 6, 5.0, {3, 3});

    // The closest signal is at (4, 4)
    EXPECT_EQ(pos.x, 4);
    EXPECT_EQ(pos.y, 4);
}