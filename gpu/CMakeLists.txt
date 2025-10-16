enable_language(CUDA)
find_package(CUDA 10.0 REQUIRED)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

### BUILD DISTRIBUTED-PBB
if(TARGET dbb)
    set(CppSources
        ${CMAKE_CURRENT_SOURCE_DIR}/worker_gpu.cpp
    )
    set(CuSources
        ${CMAKE_CURRENT_SOURCE_DIR}/gpubb.cu
    )
    message(STATUS "add to dbb target" ${CppSources})

    target_compile_options(dbb PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread>)
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)

    #set_property(TARGET Threads::Threads PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -pthread>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>")
    # set_property(TARGET Threads::Threads PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread> "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>")

    target_sources(dbb PUBLIC ${CppSources})
    target_sources(dbb PUBLIC ${CuSources})

    # add_definitions(-DFSP)

    target_include_directories(dbb PUBLIC ${PARENT_DIR}/gpu)
    target_include_directories(dbb PUBLIC ${PARENT_DIR}/gpu/include)

    target_include_directories(dbb PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_include_directories(dbb PUBLIC ${CMAKE_CUDA_INCLUDE_DIRS})
    # target_link_libraries(dbb PUBLIC ${CMAKE_CUDA_LIBRARIES})
    # target_link_libraries(dbb PUBLIC ${CUDA_DRIVER_LIBRARY})
else()
### BUILD GPU-PBB
    set(cuSources gpubb.cu)
    add_executable(gpubb main.cpp ${cuSources})

    set_target_properties(gpubb PROPERTIES CUDA_ARCHITECTURES "60;75;80;89")
    target_compile_options(gpubb PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread>)

    # set_property(TARGET Threads::Threads PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread> $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>)
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
    message(STATUS "gpubb CMAKE_CUDA_INCLUDE_DIRS " ${CMAKE_CUDA_INCLUDE_DIRS})
    message(STATUS "gpubb CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # target_link_libraries(gpubb PUBLIC ${CUDA_DRIVER_LIBRARIES})
    # target_link_libraries(gpubb PUBLIC ${CUDA_CUDART_LIBRARY})
#    target_link_libraries(gpubb PUBLIC ${PARENT_DIR}/evaluation/build/libbounds.a)
    target_link_libraries(gpubb PUBLIC bounds)
    target_link_libraries(gpubb PUBLIC fspHeuristic)

    add_executable(test test.cu)
    target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread>)
    target_include_directories(test PUBLIC ${PARENT_DIR}/gpu/include)
    target_include_directories(test PUBLIC ${CMAKE_CUDA_INCLUDE_DIRS})
    target_include_directories(test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

add_definitions(-DFSP)

if(ENUM)
    remove_definitions(-DFSP)
    add_definitions(-DTEST)
endif()
