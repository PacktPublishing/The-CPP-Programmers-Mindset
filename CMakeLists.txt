cmake_minimum_required(VERSION 3.28)

project(computational-thinking-with-cpp
        VERSION 1.0
        DESCRIPTION "Computational-Thinking-with-CPP"
        HOMEPAGE_URL "https://github.com/PacktPublishing/Computational-Thinking-with-CPP"
        LANGUAGES CXX C)

message("")
file(GLOB VAR_GLOB_DIRS LIST_DIRECTORIES true "Chapter-*")
foreach (item ${VAR_GLOB_DIRS})
    if (IS_DIRECTORY ${item} AND EXISTS "${item}/CMakeLists.txt")
        message("add_subdirectory: ${item}")
        add_subdirectory(${item})
        if ("${item}" STREQUAL "${CMAKE_SOURCE_DIR}/Chapter-04")
            break()
        endif ()
    endif ()
endforeach ()
message("")
