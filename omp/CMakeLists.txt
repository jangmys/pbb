### Check the CMake version
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

### Define project
project("OpenMP Parallel BB"
    VERSION 1.0.0
    DESCRIPTION "Parallel Branch-and-Bound"
LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-function -Wno-unused-variable -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -O0 -rdynamic" CACHE STRING "")

find_package(OpenMP REQUIRED)
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

add_executable(bb main.cpp)
add_executable(bb_seq main_noomp.cpp)


### BB OpenMP
FILE(GLOB MyCppSources ./src/*.cpp)

target_sources(bb PUBLIC ${MyCppSources})
target_sources(bb PUBLIC ${PARENT_DIR}/common/src/arguments.cpp)

target_include_directories(bb PUBLIC ./include)
target_include_directories(bb PUBLIC ${PARENT_DIR}/evaluation)
target_include_directories(bb PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(bb PUBLIC ${PARENT_DIR}/common/src)

### BB_sequential
target_sources(bb_seq PUBLIC ${MyCppSources})
target_sources(bb_seq PUBLIC ${PARENT_DIR}/common/src/arguments.cpp)

target_include_directories(bb_seq PUBLIC ./include)
target_include_directories(bb_seq PUBLIC ${PARENT_DIR}/evaluation)
target_include_directories(bb_seq PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(bb_seq PUBLIC ${PARENT_DIR}/common/src)

target_link_libraries(bb PUBLIC OpenMP::OpenMP_CXX)

link_directories(${PARENT_DIR}/build/evaluation)
# target_link_libraries(bb PUBLIC bounds)
target_link_libraries(bb PUBLIC ${PARENT_DIR}/build/evaluation/libbounds.a)
target_link_libraries(bb_seq PUBLIC ${PARENT_DIR}/build/evaluation/libbounds.a)


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(bb PUBLIC Threads::Threads)
target_link_libraries(bb_seq PUBLIC Threads::Threads)
