### Check the CMake version
# cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
#
# ### Define project
# project("PBB"
#     VERSION 1.0.0
#     DESCRIPTION "Branch-and-Bound for permutation problems"
# LANGUAGES C CXX)
#
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-function -Wno-unused-variable -O2")
set(ROOT_DIR ../..)

set(outSources
    ${ROOT_DIR}/common/src/arguments.cpp
    ${ROOT_DIR}/common/src/subproblem.cpp
)


add_executable(testHeuristic testHeuristic.cpp ${outSources})

target_include_directories(testHeuristic PUBLIC ${ROOT_DIR}/common/include)
target_include_directories(testHeuristic PUBLIC ${ROOT_DIR}/heuristics)

# target_include_directories(testHeuristic PUBLIC ${ROOT_DIR}/common/include)

target_link_libraries(testHeuristic PUBLIC fspHeuristic)

message(STATUS ${ROOT_DIR})
