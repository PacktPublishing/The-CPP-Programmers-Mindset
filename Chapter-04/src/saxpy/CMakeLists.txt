add_library(saxpy
        saxpy.cpp
        saxpy_hand.cpp
)

target_include_directories(saxpy PRIVATE include)
target_include_directories(saxpy PUBLIC include/saxpy)

target_link_Libraries(saxpy PUBLIC my_cpuid)

if (MSVC)
    set(ARCH_FLAG "/arch:")
else ()
    set(ARCH_FLAG "-m")
endif ()

set_property(SOURCE saxpy_hand.cpp APPEND PROPERTY
        COMPILE_OPTIONS ${ARCH_FLAG}avx2
)

set(INST default)
configure_file(saxpy.cpp.in saxpy_${INST}.cpp @ONLY)
target_sources(saxpy PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/saxpy_${INST}.cpp)
foreach (ARCH IN ITEMS sse4.2 avx2 avx512f)
    string(REPLACE "." "" INST ${ARCH})
    configure_file(saxpy.cpp.in saxpy_${INST}.cpp @ONLY)
    set_property(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/saxpy_${INST}.cpp"
            APPEND PROPERTY COMPILE_OPTIONS "${ARCH_FLAG}${ARCH}")
    target_sources(saxpy PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/saxpy_${INST}.cpp)
endforeach ()
