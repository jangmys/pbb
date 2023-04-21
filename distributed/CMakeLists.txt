set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(MPI REQUIRED)

if(GPU)
    enable_language(CUDA)
    add_definitions(-DFSP)
endif()


get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

add_executable(dbb src/main.cpp)

add_subdirectory(src)
add_subdirectory(${PARENT_DIR}/common/src ./build_common)
add_subdirectory(${PARENT_DIR}/multicore/base ./build_mcbase)
add_subdirectory(${PARENT_DIR}/multicore/ivm ./build_mcivm)
add_subdirectory(${PARENT_DIR}/multicore/do ./build_mcdo)
add_subdirectory(${PARENT_DIR}/multicore/ll ./build_mcll)
add_subdirectory(${PARENT_DIR}/multicore/operators ./build_mcoperators)

if(GPU)
    add_subdirectory(${PARENT_DIR}/gpu ./build_gpu)
endif()


if(GPU)
    target_include_directories(dbb PUBLIC ${PARENT_DIR}/gpu)
    target_include_directories(dbb PUBLIC ${PARENT_DIR}/gpu/include)

    # set_target_properties(dbb PROPERTIES CUDA_ARCHITECTURES "35;50;72")
    # set_property(TARGET dbb PROPERTY CUDA_ARCHITECTURES OFF)
    #
    # target_link_libraries(dbb PUBLIC ${CMAKE_CUDA_LIBRARIES})
    # target_link_libraries(dbb PUBLIC ${CUDA_DRIVER_LIBRARY})
    # target_link_libraries(dbb PUBLIC ${CUDA_CUDART_LIBRARY})
endif()

target_include_directories(dbb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(dbb PUBLIC ${PARENT_DIR}/evaluation)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/common/src)

target_include_directories(dbb PUBLIC ${PARENT_DIR}/heuristics)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/multicore/operators)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/multicore/base)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/multicore/ivm)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/multicore/do)
target_include_directories(dbb PUBLIC ${PARENT_DIR}/multicore/ll)

target_include_directories(dbb PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(dbb PUBLIC ${MPI_CXX_INCLUDE_PATH})

target_compile_options(dbb PUBLIC ${MPI_CXX_COMPILE_FLAGS})

target_link_libraries(dbb PUBLIC Threads::Threads)
target_link_libraries(dbb PUBLIC bounds)
target_link_libraries(dbb PUBLIC fspHeuristic)
target_link_libraries(dbb PUBLIC gmp gmpxx)
target_link_libraries(dbb PUBLIC MPI::MPI_CXX)

if(GPU)
    set_target_properties(dbb PROPERTIES CUDA_ARCHITECTURES "60;72")
    set_property(TARGET dbb PROPERTY CUDA_ARCHITECTURES OFF)

    target_link_libraries(dbb PUBLIC ${CMAKE_CUDA_LIBRARIES})
    target_link_libraries(dbb PUBLIC ${CUDA_DRIVER_LIBRARY})
    target_link_libraries(dbb PUBLIC ${CUDA_CUDART_LIBRARY})
endif()
