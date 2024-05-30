### Check the CMake version
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

### Define project
project("PBB"
    VERSION 1.0.0
    DESCRIPTION "Branch-and-Bound for permutation problems"
LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O2 -DNDEBUG")#-Wno-unused-function -Wno-unused-variable -O2 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-std=c99 -Wall -Wextra -pedantic -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -O0 -rdynamic" CACHE STRING "")

#if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x64_86")
#endif()

if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "build for aarch64")
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mcpu=neoverse-n1")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=neoverse-n1")
endif()

######################################################################################

#optional : add GPUBB (requires CUDA)
if(GPU)
    add_definitions(-DWITH_GPU)
    add_compile_definitions(WITH_GPU)
    add_subdirectory(gpu)
endif()

#optional : add distributed BB (requires MPI and GMP)
if(MPI)
    add_definitions(-DWITH_MPI)
    add_compile_definitions(WITH_MPI)
    add_subdirectory(distributed)
endif()




#add lower bounds library
add_subdirectory(evaluation)
#add heuristics library
add_subdirectory(heuristics)
#add multicore IVM-based BB
add_subdirectory(multicore)

######################################################################################

unset(CMAKE_BUILD_TYPE CACHE)
