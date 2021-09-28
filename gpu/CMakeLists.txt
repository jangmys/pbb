### Check the CMake version
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

### Define project
project(gpubb
    VERSION 1.0.0
    DESCRIPTION "GPU-based Branch-and-Bound for permutation problems"
    LANGUAGES CUDA CXX)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-function -O2")
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA 10.0 REQUIRED)
#for CMake <3.17

# Do what the new package does
find_library(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
	     HINTS ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}
	     ENV CMAKE_CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
if (NOT CUDA_DRIVER_LIBRARY)
    # Don't try any stub directories until we have exhausted all other search locations.
    find_library(CUDA_DRIVER_LIBRARY
                 NAMES cuda_driver cuda
		 HINTS ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}
		 ENV CMAKE_CUDA_PATH
                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs)
endif ()
mark_as_advanced(CUDA_DRIVER_LIBRARY)

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)
# message(STATUS "cudalib" ${CUDA_LIBRARIES})

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

file(GLOB cuSources gpubb.cu)
add_executable(gpubb main.cpp ${cuSources})

add_definitions(-DFSP)

add_subdirectory(${PARENT_DIR}/common/src ./build_common_gpu)
target_include_directories(gpubb PUBLIC ${PARENT_DIR}/evaluation)
target_include_directories(gpubb PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(gpubb PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(gpubb PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(gpubb PUBLIC ${CMAKE_CUDA_INCLUDE_DIRS})
target_link_libraries(gpubb PUBLIC ${CMAKE_CUDA_LIBRARIES})
target_link_libraries(gpubb PUBLIC ${CUDA_DRIVER_LIBRARY})

if(TARGET dbb)
    FILE(GLOB CppSources *.cpp)
    target_sources(dbb PRIVATE ${CppSources})
endif()
