cmake_minimum_required(VERSION 3.30)
project(ct6)

set(CMAKE_CXX_STANDARD 20)

add_library(MySharedLibrary SHARED library.cpp)

set_target_properties(MySharedLibrary PROPERTIES
        VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

include(GenerateExportHeader)

generate_export_header(MySharedLibrary)

target_include_directories(MySharedLibrary PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR})


add_library(MyModuleExample STATIC
        exported_fns.cpp)

target_sources(MyModuleExample PRIVATE FILE_SET modules TYPE CXX_MODULES FILES module_example.ixx)


add_subdirectory(ctode)