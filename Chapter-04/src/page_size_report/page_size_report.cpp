//
// Created by sammorley on 01/01/25.
//


#include <iostream>

/*
 * This example follows the code example from the Wikipedia page on memory pages
 * https://en.wikipedia.org/w/index.php?title=Page_(computer_memory)&oldid=1250593554
 */
#ifdef __linux__
#include <unistd.h>
#elif _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

int main() {
#ifdef __linux__
    std::cout << "page size: " << sysconf(_SC_PAGESIZE)  << 'B' << std::endl;
#elif __WIN32__
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    std::cout << "page size: " << info.dwPageSize << 'B' << std::endl;
#endif

    return 0;
}