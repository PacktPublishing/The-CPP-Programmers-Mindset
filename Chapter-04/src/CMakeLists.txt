#add_subdirectory(app1)
#add_subdirectory(lib2)

###############################################################################
# Multithreading
###############################################################################

# We've separated out the implementations so we can write some benchmarks
add_subdirectory(open_mp_demo)
add_subdirectory(bench_nearest_neighbor)
add_subdirectory(nearest_neighbor_functions)
