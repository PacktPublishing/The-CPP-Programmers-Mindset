

add_library(duckies_common INTERFACE duckies_types.h)

target_include_directories(duckies_common INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
)
