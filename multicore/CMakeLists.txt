### Check the CMake version
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

### Define project
project(bounds
    VERSION 1.0.0
    DESCRIPTION "Branch-and-Bound for permutation problems"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-function -O2")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)


get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)


add_executable(bb main.cpp)
add_subdirectory(base)
add_subdirectory(operators)
add_subdirectory(ivm)
# add_subdirectory(src)
add_subdirectory(${PARENT_DIR}/common/src ./build_common_mc)
    # add_subdirectory(${PARENT_DIR}/common/src ./build_common)
    # add_subdirectory("${CMAKE_SOURCE_DIR}/common/libplugin")
# endif()

target_include_directories(bb PUBLIC ${PARENT_DIR}/evaluation)

target_include_directories(bb PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(bb PUBLIC ${PARENT_DIR}/common/src)

target_include_directories(bb PUBLIC ${PROJECT_SOURCE_DIR}/base/)
target_include_directories(bb PUBLIC ${PROJECT_SOURCE_DIR}/ivm/)
target_include_directories(bb PUBLIC ${PROJECT_SOURCE_DIR}/operators/)
# target_include_directories(bb PUBLIC ${PROJECT_SOURCE_DIR}/include/)
# target_include_directories(bb PUBLIC ${PROJECT_SOURCE_DIR}/src/)

message(STATUS ${SOURCES})

target_link_libraries(bb PUBLIC Threads::Threads)
target_link_libraries(bb PUBLIC bounds)
# target_link_libraries(bb PUBLIC common)

# target_link_libraries(bb PRIVATE ${PARENT_DIR}/evaluation/lib/libbounds.a)
