#include "max_element.hpp"


#include <gtest/gtest.h>


#include <vector>


TEST(MaxElement, TestMaxListInts) {
    std::vector<int> v{1, 2, 3, 4, 5};

    auto max = ct::max_element(v);

    EXPECT_EQ(max, 5);
}


TEST(MaxElementConcept, TestMaxListInts) {
    std::vector<int> v{1, 2, 3, 4, 5};

    auto max = ct::derived_concept::max_element(v);

    EXPECT_EQ(max, 5);
}