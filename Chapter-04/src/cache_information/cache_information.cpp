#include <array>
#include <iostream>
#include <string>


#include "my_cpuid.h"





int main() {
    unsigned cpuid_registers[4];

    my_cpuid(0, 0, cpuid_registers);
    unsigned max_supported_function = cpuid_registers[0];

    if (max_supported_function < 4) {
        return 1;
    }

    std::string vendor;
    vendor.reserve(12);
    vendor.append(reinterpret_cast<const char *>(cpuid_registers + 1), 4);
    vendor.append(reinterpret_cast<const char *>(cpuid_registers + 3), 4);
    vendor.append(reinterpret_cast<const char *>(cpuid_registers + 2), 4);

    unsigned cache_leaf = 0x04;
    if (vendor == "AuthenticAMD") {
        cache_leaf = 0x8000'001D;
    }

    static const std::array<std::string, 4> cache_types = {
        "Null", "Data", "Instruction", "Unified"
    };

    for (unsigned i = 0; i < 6; ++i) { // Iterate through available cache levels
        my_cpuid(cache_leaf, i, cpuid_registers);

        // EAX
        // Bits 4:0 (0 = Null, 1 = Data, 2 = Instruction, 3 = Unified)
        unsigned type = cpuid_registers[0] & 0x1F;
        // If the type is zero, there are no more cache entries
        if (type == 0) break;

        // Bits 7:5 - Cache level
        unsigned level = (cpuid_registers[0] >> 5) & 0x7;

        // EBX
        // Bits 11:0 - Line size - 1
        unsigned line_size = (cpuid_registers[1] & 0xFFF) + 1;
        // Bits 21:12 - Partitions - 1
        unsigned partitions = ((cpuid_registers[1] >> 12) & 0x3FF) + 1;
        // Bits 31:22 - Ways of associativity - 1
        unsigned associativity = ((cpuid_registers[1] >> 22) & 0x3FF) + 1;

        // ECX
        // Number of sets - 1
        unsigned sets = cpuid_registers[2] + 1;

        // Cache size in bytes = line size × partitions × associativity × sets
        unsigned size = line_size * partitions * associativity * sets;

        std::cout << "L" << level << " " << cache_types[type]
                  << ": " << size / 1024 << " KiB" << std::endl;
    }

    return 0;
}