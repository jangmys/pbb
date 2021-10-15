set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

message(STATUS "DIR " ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bb main.cpp)
add_subdirectory(base)
add_subdirectory(operators)
add_subdirectory(ivm)
add_subdirectory(${PARENT_DIR}/common/src ./build_common_mc)

target_include_directories(bb PUBLIC ${PARENT_DIR}/evaluation)
target_include_directories(bb PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(bb PUBLIC ${PARENT_DIR}/common/src)
target_include_directories(bb PUBLIC ${PARENT_DIR}/heuristics)

target_include_directories(bb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/base)
target_include_directories(bb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ivm)
target_include_directories(bb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/operators)

target_link_libraries(bb PUBLIC Threads::Threads)
target_link_libraries(bb PUBLIC bounds)
target_link_libraries(bb PUBLIC fspHeuristic)
