### Check the CMake version
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

### Define project
project(main
    VERSION 1.0.0
    DESCRIPTION "Branch-and-Bound for permutation problems"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-function -O2")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(MPI REQUIRED)
# find_package(GMP REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)


message(STATUS ${PARENT_DIR})

add_executable(dbb src/main.cpp)
add_subdirectory(src)
# add_subdirectory(operators)
# add_subdirectory(ivm)
# add_subdirectory(src)
add_subdirectory(${PARENT_DIR}/common/src ./build_common)
add_subdirectory(${PARENT_DIR}/multicore/base ./build_mcbase)
add_subdirectory(${PARENT_DIR}/multicore/ivm ./build_mcivm)
# if(GPU)
#     add_subdirectory(${PARENT_DIR}/gpu ./build_gpu)
# endif()

    # add_subdirectory(${PARENT_DIR}/common/src ./build_common)
    # add_subdirectory("${CMAKE_SOURCE_DIR}/common/libplugin")
# endif()

# message(STATUS ${PROJECT_SOURCE_DIR})

target_include_directories(dbb PUBLIC ${PARENT_DIR}/evaluation)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/common/src)

target_include_directories(dbb PUBLIC ${PARENT_DIR}/multicore/operators)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/multicore/base)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/multicore/ivm)
if(GPU)
    target_include_directories(dbb PUBLIC ${PARENT_DIR}/gpu)
endif()


target_include_directories(dbb PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(dbb PUBLIC ${MPI_CXX_INCLUDE_PATH})

target_compile_options(dbb PUBLIC ${MPI_CXX_COMPILE_FLAGS})
# target_include_directories(bb PUBLIC ${PROJECT_SOURCE_DIR}/ivm/)
# target_include_directories(bb PUBLIC ${PROJECT_SOURCE_DIR}/operators/)
# target_include_directories(bb PUBLIC ${PROJECT_SOURCE_DIR}/include/)
# target_include_directories(bb PUBLIC ${PROJECT_SOURCE_DIR}/src/)

# message(STATUS "The sources" ${SOURCES})

target_link_libraries(dbb PUBLIC Threads::Threads)
target_link_libraries(dbb PUBLIC bounds)
target_link_libraries(dbb PUBLIC gmp gmpxx)
target_link_libraries(dbb PUBLIC MPI::MPI_CXX)
# target_link_libraries(bb PUBLIC common)

# target_link_libraries(bb PRIVATE ${PARENT_DIR}/evaluation/lib/libbounds.a)
