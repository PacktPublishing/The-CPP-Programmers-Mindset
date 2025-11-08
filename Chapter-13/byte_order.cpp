#include <iostream>
#include <iomanip>
#include <array>
#include <cstring>
#include <cstdint>


int main() {
    int num = 0x1A'2B'3C'4D;
    std::array<std::byte, sizeof(int)> byte_array{};
    std::memcpy(byte_array.data(), &num, sizeof(int));

    std::cout << "0x";
    for (auto byte: byte_array) {
        std::cout << std::setw(2) << std::hex
                << static_cast<unsigned>(byte);
    }
    std::cout << std::endl;

    return 0;
}