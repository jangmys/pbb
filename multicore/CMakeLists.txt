set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)


add_executable(bb ivm/main.cpp)
add_subdirectory(ivm)

add_executable(llbb ll/main.cpp)
add_subdirectory(ll)

add_subdirectory(base)
add_subdirectory(operators)
# add_subdirectory(do)
add_subdirectory(${PARENT_DIR}/common/src ./build_common_mc)

target_include_directories(bb PUBLIC ${PARENT_DIR}/evaluation)
target_include_directories(bb PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(bb PUBLIC ${PARENT_DIR}/common/src)
target_include_directories(bb PUBLIC ${PARENT_DIR}/heuristics)

target_include_directories(bb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/base)
# target_include_directories(bb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/do)
target_include_directories(bb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ivm)
# target_include_directories(bb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ll)
target_include_directories(bb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/operators)

target_link_libraries(bb PUBLIC Threads::Threads)
target_link_libraries(bb PUBLIC bounds)
target_link_libraries(bb PUBLIC fspHeuristic)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/base ./build_base_mc)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/operators ./build_operators_mc)
# add_subdirectory(${PARENT_DIR}/common/src ./build_common_mcll)

target_include_directories(llbb PUBLIC ${PARENT_DIR}/evaluation)
target_include_directories(llbb PUBLIC ${PARENT_DIR}/common/include)
target_include_directories(llbb PUBLIC ${PARENT_DIR}/common/src)
target_include_directories(llbb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/base)
target_include_directories(llbb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/operators)
# target_include_directories(llbb PUBLIC ${PARENT_DIR}/multicore/base)
# target_include_directories(llbb PUBLIC ${PARENT_DIR}/multicore/operators)
target_include_directories(llbb PUBLIC ${PARENT_DIR}/heuristics)

target_link_libraries(llbb PUBLIC Threads::Threads)
target_link_libraries(llbb PUBLIC bounds)
target_link_libraries(llbb PUBLIC fspHeuristic)
