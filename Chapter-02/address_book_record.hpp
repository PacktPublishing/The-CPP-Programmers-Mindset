#ifndef CHAPTER_02_ADDRESS_BOOK_RECORD_HPP
#define CHAPTER_02_ADDRESS_BOOK_RECORD_HPP


#include <cstddef>
#include <string>

namespace ct {
// This struct does not have optimal layout, don't worry about it

struct AddressBookRecord {
    size_t id;
    int house_number;
    std::string street_address;
    std::string city_and_state;
    int zip_code;
};

inline bool
operator==(const AddressBookRecord &lhs, const AddressBookRecord &rhs)
    noexcept {
    return lhs.id == rhs.id;
}

bool compare_house_number(const AddressBookRecord &left, const AddressBookRecord &right);

bool compare_zip_code(const AddressBookRecord &left, const AddressBookRecord &right);


bool operator<(const AddressBookRecord &lhs, const AddressBookRecord &rhs);


class RecordView {
    const AddressBookRecord *p_data;

public:
    // deliberately non-explicit constructor
    constexpr RecordView(const AddressBookRecord &data) : p_data(&data) {
    }

    [[nodiscard]]
    constexpr size_t id() const noexcept { return p_data->id; }

    [[nodiscard]]
    constexpr int house_number() const noexcept { return p_data->house_number; }

    [[nodiscard]]
    constexpr const std::string &street_address() const noexcept { return p_data->street_address; }

    [[nodiscard]]
    constexpr const std::string &city_and_state() const noexcept { return p_data->city_and_state; }

    [[nodiscard]]
    constexpr int zip_code() const noexcept { return p_data->zip_code; }
};

inline bool operator==(const RecordView &lhs, const RecordView &rhs) noexcept {
    return lhs.id() == rhs.id();
}
} // namespace ct


#endif //CHAPTER_02_ADDRESS_BOOK_RECORD_HPP