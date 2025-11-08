#ifndef CHAPTER_01_RECURSIVE_WORD_HPP
#define CHAPTER_01_RECURSIVE_WORD_HPP


#include <cstdint>
#include <string_view>


namespace ct {
namespace recursive {
size_t end_of_first_word(std::string_view s) noexcept;
} // namespace recursive

namespace non_recursive {
size_t end_of_first_word(std::string_view s) noexcept;
} // namespace non_recursive


namespace ranges {
size_t end_of_first_word(std::string_view s) noexcept;
} // namespace ranges
}

#endif //CHAPTER_01_RECURSIVE_WORD_HPP