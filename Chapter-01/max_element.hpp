#ifndef CHAPTER_01_MAX_ELEMENT_HPP
#define CHAPTER_01_MAX_ELEMENT_HPP

#include <stdexcept>
#include <concepts>
#include <ranges>


/*
The following code is a direct implementation of the following
algorithm. It is not supposed to be the most efficient, or
standards compliant version of this algorithm. The only purpose
is to translate directly from algorithm to code. This makes
several unreasonable assumptions on the Container::value_type
type, including that it is copy constructible/assignable.
Moreover, it is probably not fast since it must copy the,
potentially large value_type on every iteration. Don't use this
code, use std::max_element instead.

INPUT: L is a list of numbers with at least one element
OUTPUT: Maximum value of L
max <- first element of L

WHILE not at end of L
    current <- next element of L
    IF current > max
        max <- current
    END
END

RETURN max
*/


namespace ct {
template<typename Container>
typename Container::value_type max_element(const Container &container) {
    auto begin = container.begin();
    auto end = container.end();
    if (begin == end) {
        throw std::invalid_argument("container must be non-empty");
    }
    auto max = *begin;
    ++begin;

    for (; begin != end; ++begin) {
        const auto &current = *begin;
        if (max < current) {
            max = current;
        }
    }
    return max;
}


namespace basic_concept {
template<typename T>
concept OrderableContainer = requires(const T &t)
{
    // Has a dependent type called "value_type", which is ordered by <
    requires std::totally_ordered<typename T::value_type>;

    // Has a begin and end method valid on const T&
    t.begin();
    t.end();
};
} // namespace basic_concept


namespace derived_concept {
template<typename T>
concept OrderableContainer = std::ranges::input_range<const T>
                             && std::totally_ordered<std::ranges::range_value_t<const T> >
                             && std::copy_constructible<std::ranges::range_value_t<const T> >;


template<OrderableContainer Container>
std::ranges::range_value_t<Container> max_element(const Container &container) {
    auto begin = std::ranges::begin(container);
    const auto end = std::ranges::end(container);
    if (begin == end) {
        throw std::invalid_argument("container must be non-empty");
    }

    auto max = *begin;
    ++begin;

    for (; begin != end; ++begin) {
        const auto &current = *begin;
        if (max < current) {
            max = current;
        }
    }

    return max;
}
}
} // namespace ct


#endif //CHAPTER_01_MAX_ELEMENT_HPP