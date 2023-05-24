enable_language(CUDA)
find_package(CUDA 10.0 REQUIRED)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

if(TARGET dbb)
    set(CppSources
        ${CMAKE_CURRENT_SOURCE_DIR}/worker_gpu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpubb.cu
    )

    message(STATUS "add to dbb target" ${CppSources})

    target_sources(dbb PUBLIC ${CppSources})

    # add_definitions(-DFSP)

    target_include_directories(dbb PUBLIC ${PARENT_DIR}/gpu)
    target_include_directories(dbb PUBLIC ${PARENT_DIR}/gpu/include)

    target_include_directories(dbb PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_include_directories(dbb PUBLIC ${CMAKE_CUDA_INCLUDE_DIRS})
    target_include_directories(dbb PUBLIC ${CMAKE_CUDA_INCLUDE_DIRS})
    # target_link_libraries(dbb PUBLIC ${CMAKE_CUDA_LIBRARIES})
    # target_link_libraries(dbb PUBLIC ${CUDA_DRIVER_LIBRARY})
else()
    set(cuSources gpubb.cu)
    add_executable(gpubb main.cpp ${cuSources})

    set_target_properties(gpubb PROPERTIES CUDA_ARCHITECTURES "60;72")

    set_property(TARGET Threads::Threads PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread> $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>)

    # add_definitions(-DFSP)

    add_subdirectory(${PARENT_DIR}/common/src ./build_common_gpu)

    target_include_directories(gpubb PUBLIC ${PARENT_DIR}/gpu/include)
    target_include_directories(gpubb PUBLIC ${PARENT_DIR}/evaluation)
    target_include_directories(gpubb PUBLIC ${PARENT_DIR}/heuristics)
    target_include_directories(gpubb PUBLIC ${PARENT_DIR}/multicore)
    target_include_directories(gpubb PUBLIC ${PARENT_DIR}/common/include)
    target_include_directories(gpubb PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(gpubb PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    target_include_directories(gpubb PUBLIC ${CMAKE_CUDA_INCLUDE_DIRS})

    message(STATUS "CMAKE_CUDA" ${CMAKE_CUDA_LIBRARIES})
    message(STATUS "gpubb CUDA_DRIVER" ${CUDA_DRIVER_LIBRARY})
    message(STATUS "gpubb CUDART " ${CUDA_CUDART_LIBRARY})

    target_link_libraries(gpubb PUBLIC ${CUDA_DRIVER_LIBRARIES})
    target_link_libraries(gpubb PUBLIC ${CUDA_CUDART_LIBRARY})
#    target_link_libraries(gpubb PUBLIC ${PARENT_DIR}/evaluation/build/libbounds.a)
    target_link_libraries(gpubb PUBLIC bounds)
    target_link_libraries(gpubb PUBLIC fspHeuristic)
endif()

add_definitions(-DFSP)

if(ENUM)
    remove_definitions(-DFSP)
    add_definitions(-DTEST)
endif()
