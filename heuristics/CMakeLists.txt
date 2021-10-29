set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

add_library(fspHeuristic) # ${MyCppSources})
add_subdirectory(flowshop)
add_subdirectory(${PARENT_DIR}/common/src ./build_common_heulib)

target_include_directories(fspHeuristic PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(fspHeuristic PUBLIC ${PARENT_DIR}/multicore/operators)
target_include_directories(fspHeuristic PUBLIC ${PARENT_DIR}/evaluation)

target_include_directories(fspHeuristic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)
target_include_directories(fspHeuristic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/neh)
target_include_directories(fspHeuristic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/localsearch)
target_include_directories(fspHeuristic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/beam)

target_link_libraries(fspHeuristic PUBLIC ${PARENT_DIR}/build/evaluation/libbounds.a)

add_subdirectory(test)
