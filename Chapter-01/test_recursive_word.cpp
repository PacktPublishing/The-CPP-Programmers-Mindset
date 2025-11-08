#include "recursive_word.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace ct;


TEST(RecursiveWord, TestRecursiveOneLetterWord) {
    std::string word = "a";

    auto end = recursive::end_of_first_word(word);

    EXPECT_EQ(end, 0);
}

TEST(RecursiveWord, TestNonRecursiveOneLetterWord) {
    std::string word = "a";

    auto end = non_recursive::end_of_first_word(word);

    EXPECT_EQ(end, 0);
}


TEST(RecursiveWord, TestRecursiveOneBracketWord) {
    std::string word = "[a,a]";

    auto end = recursive::end_of_first_word(word);

    EXPECT_EQ(end, 4);
}

TEST(RecursiveWord, TestNonRecursiveOneBracketWord) {
    std::string word = "[a,a]";

    auto end = non_recursive::end_of_first_word(word);

    EXPECT_EQ(end, 4);
}

TEST(RecursiveWord, TestRecursiveNestedBracketWord) {
    std::string word = "[a,[a,a]]";

    auto end = recursive::end_of_first_word(word);

    EXPECT_EQ(end, 8);
}

TEST(RecursiveWord, TestNonRecursiveNestedBracketWord) {
    std::string word = "[a,[a,b]]";

    auto end = non_recursive::end_of_first_word(word);

    EXPECT_EQ(end, 8);
}

TEST(RecursiveWord, TestRecursiveLongWord) {
    std::string word = "[[a,a],[a,[a,b]]]";

    auto end = recursive::end_of_first_word(word);

    EXPECT_EQ(end, 16);
}

TEST(RecursiveWord, TestNonRecursiveLongWord) {
    std::string word = "[[a,a],[a,[a,b]]]";

    auto end = non_recursive::end_of_first_word(word);

    EXPECT_EQ(end, 16);
}


TEST(RecursiveWord, TestRangesOneLetterWord) {
    std::string word = "a";

    auto end = ranges::end_of_first_word(word);

    EXPECT_EQ(end, 0);
}

TEST(RecursiveWord, TestRangesOneBracketWord) {
    std::string word = "[a,a]";

    auto end = ranges::end_of_first_word(word);

    EXPECT_EQ(end, 4);
}

TEST(RecursiveWord, TestRangesNestedBracketWord) {
    std::string word = "[a,[a,b]]";

    auto end = ranges::end_of_first_word(word);

    EXPECT_EQ(end, 8);
}

TEST(RecursiveWord, TestRangesLongWord) {
    std::string word = "[[a,a],[a,[a,b]]]";

    auto end = ranges::end_of_first_word(word);

    EXPECT_EQ(end, 16);
}