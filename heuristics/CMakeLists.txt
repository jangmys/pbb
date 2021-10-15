set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

message(STATUS "DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "DIR: " ${PROJECT_SOURCE_DIR})

# file(GLOB MyCppSources
#         ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/neh/*.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/ils/*.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/localsearch/*.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/beam/*.cpp
#     )

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

target_link_libraries(fspHeuristic ${PARENT_DIR}/build/evaluation/libbounds.a)



# add_executable(testHeuristic test/testHeuristic.cpp)
#
# add_subdirectory(${PARENT_DIR}/common/src ./build_common_heu)
#
# target_include_directories(testHeuristic PUBLIC ${PARENT_DIR}/common/include)
# target_include_directories(testHeuristic PUBLIC ${PARENT_DIR}/common/src)
#
# target_include_directories(testHeuristic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)
# target_include_directories(testHeuristic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/neh)
# target_include_directories(testHeuristic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/localsearch)
# target_include_directories(testHeuristic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flowshop/beam)
#
# target_link_libraries(testHeuristic Threads::Threads)
# target_link_libraries(testHeuristic fspHeuristic)
