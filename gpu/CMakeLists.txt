### Check the CMake version
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

### Define project
project(gpubb
    VERSION 1.0.0
    DESCRIPTION "GPU-based Branch-and-Bound for permutation problems"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-function -O2")



if(TARGET dbb)
    FILE(GLOB CppSources *.cpp)
    target_sources(dbb PRIVATE ${CppSources})
endif()
