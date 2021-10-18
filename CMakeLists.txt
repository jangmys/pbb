### Check the CMake version
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

### Define project
project("PBB"
    VERSION 1.0.0
    DESCRIPTION "Branch-and-Bound for permutation problems"
LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-function -Wno-unused-variable -O2")

######################################################################################
### 3) Include CMake files
######################################################################################

## Configuration file for building type and flags
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake)

## Macro file
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Macro.cmake)

## Custom Targets file
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Target.cmake)

#add lower bounds library
add_subdirectory(evaluation)
#add multicore IVM-based BB
add_subdirectory(multicore)
#add heuristics library
add_subdirectory(heuristics)

#optional : add GPUBB (requires CUDA)
if(GPU)
    add_definitions(-DUSE_GPU)
    add_subdirectory(gpu)
endif()

#optional : add distributed BB (requires MPI and GMP)
if(MPI)
    add_definitions(-DWITH_MPI)
    add_subdirectory(distributed)
endif()
