#include "recursive_word.hpp"

#include <cassert>
#include <ranges>


size_t ct::recursive::end_of_first_word(std::string_view s) noexcept {
    if (!s.starts_with('[')) {
        return 0;
    }

    size_t position = 0;
    assert(s[position] == '[');

    position += 1;
    auto a = s.substr(position);
    auto i = end_of_first_word(a);
    position += i;

    position += 1;
    assert(s[position] == ',');
    position += 1;

    auto b = s.substr(position);
    auto j = end_of_first_word(b);
    position += j;

    position += 1;
    assert(s[position] == ']');

    return position;
}


size_t ct::non_recursive::end_of_first_word(std::string_view s) noexcept {
    size_t position = 0;
    int depth = 0;
    for (const auto &c: s) {
        switch (c) {
            case '[': ++depth;
                break;
            case ']': --depth;
            default:
                if (depth == 0) {
                    return position;
                }
        }
        ++position;
    }
    return position;
}


size_t ct::ranges::end_of_first_word(std::string_view s) noexcept {
    int depth = 0;
    for (const auto [index, chr]: std::views::enumerate(s)) {
        switch (chr) {
            case '[': ++depth;
                break;
            case ']': --depth;
            default:
                if (depth == 0) {
                    return index;
                }
        }
    }
    return s.size();
}