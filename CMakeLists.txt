cmake_minimum_required(VERSION 3.28)

project(computational-thinking-with-cpp
        VERSION 1.0
        DESCRIPTION "Computational-Thinking-with-CPP"
        HOMEPAGE_URL "https://github.com/PacktPublishing/Computational-Thinking-with-CPP"
        LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message("")
file(GLOB VAR_GLOB_DIRS LIST_DIRECTORIES true "Chapter-*")
foreach (item ${VAR_GLOB_DIRS})
    if (IS_DIRECTORY ${item} AND EXISTS "${item}/CMakeLists.txt")
        if ("${item}" STRLESS_EQUAL "${CMAKE_SOURCE_DIR}/Chapter-04")
            message("add_subdirectory: ${item}")
            add_subdirectory(${item} bin)
        endif ()
    endif ()
endforeach ()
message("")
