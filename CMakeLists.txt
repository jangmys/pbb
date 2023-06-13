### Check the CMake version
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

### Define project
project("PBB"
    VERSION 1.0.0
    DESCRIPTION "Branch-and-Bound for permutation problems"
LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O2 -DNDEBUG")#-Wno-unused-function -Wno-unused-variable -O2 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-std=c99 -Wall -Wextra -pedantic -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -O0 -rdynamic" CACHE STRING "")

#if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x64_86")
#endif()

if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -neoverse-n1")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -neoverse-n1")
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

if(TESTS)
    ######################################################################################
    ### googletest
    ######################################################################################
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(
        test_test
        tests/test_ivm.cpp
        multicore/ivm/ivm.cpp
    )
    target_include_directories(
        test_test
        PUBLIC
        evaluation
    )

    target_link_libraries(
        test_test
        gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(test_test)
endif()
