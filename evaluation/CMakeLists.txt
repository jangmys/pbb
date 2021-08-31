### Check the CMake version
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

### Define project
project(bounds
    VERSION 1.0.0
    DESCRIPTION "Branch-and-Bound for permutation problems"
    LANGUAGES C CXX)

# set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -O2 -Wno-unused-function")

include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/)

add_library(bounds ${SOURCES})
add_subdirectory(flowshop)
target_include_directories(bounds PUBLIC ${PROJECT_SOURCE_DIR}/)


add_executable(testBounds test/testBounds.cpp)
# target_link_libraries(testBounds ${PROJECT_SOURCE_DIR}/lib/libbounds.a)
target_link_libraries(testBounds bounds)
