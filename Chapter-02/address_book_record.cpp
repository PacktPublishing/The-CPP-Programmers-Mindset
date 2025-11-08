#include "address_book_record.hpp"

#include <algorithm>

#include <ranges>

bool ct::compare_house_number(const AddressBookRecord &left, const AddressBookRecord &right) {
    return left.house_number < right.house_number;
}

bool ct::compare_zip_code(const AddressBookRecord &left, const AddressBookRecord &right) {
    return left.zip_code < right.zip_code;
}


bool ct::operator<(const AddressBookRecord &lhs, const AddressBookRecord &rhs) {
    // Start by checking if the zip code is different. If they are not the same,
    // the order is determined completely by the zip code.
    if (lhs.zip_code != rhs.zip_code) {
        return lhs.zip_code < rhs.zip_code;
    }

    // Next is city then state. Again, if these are not equal then it determines
    // the order. These are compared lexicographically
    if (!std::ranges::equal(lhs.city_and_state, rhs.city_and_state)) {
        return std::ranges::lexicographical_compare(lhs.city_and_state, rhs.city_and_state);
    }

    // Street address is next, same pattern again.
    if (!std::ranges::equal(lhs.street_address, rhs.street_address)) {
        return std::ranges::lexicographical_compare(lhs.street_address, rhs.street_address);
    }

    // If we're here, then we're on the same street. So house number is all that
    // separates addresses.
    return lhs.house_number < rhs.house_number;
}