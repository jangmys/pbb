### Check the CMake version
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

### Define project
project(gpubb
    VERSION 1.0.0
    DESCRIPTION "GPU-based Branch-and-Bound for permutation problems"
    LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-function -O2")

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

add_executable(gpubb main.cpp)

message(STATUS "blabla" ${CUDA_INCLUDE_DIRS})
message(STATUS "blabla" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(${PARENT_DIR}/common/src ./build_common_gpu)
target_include_directories(gpubb PUBLIC ${PARENT_DIR}/evaluation)
target_include_directories(gpubb PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(gpubb PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(gpubb PUBLIC ${PROJECT_SOURCE_DIR}/include)


if(TARGET dbb)
    FILE(GLOB CppSources *.cpp)
    target_sources(dbb PRIVATE ${CppSources})
endif()