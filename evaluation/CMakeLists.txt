set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(bounds)

target_sources(bounds PRIVATE instance_factory.cpp bound_abstract.cpp)

add_subdirectory(${PROJECT_SOURCE_DIR}/evaluation/flowshop)
add_subdirectory(${PROJECT_SOURCE_DIR}/evaluation/dummy)
target_include_directories(bounds PUBLIC ${PROJECT_SOURCE_DIR}/evaluation)


target_link_libraries(bounds PUBLIC Threads::Threads)

# install(TARGETS bounds ${PROJECT_SOURCE_DIR}/lib/)
# install(TARGETS bounds LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib/ COMPONENT libraries)

add_executable(testBounds test/testBounds.cpp)
target_link_libraries(testBounds bounds)

add_executable(ctest test/c_test.c)
target_include_directories(ctest PUBLIC ${PROJECT_SOURCE_DIR}/evaluation/flowshop/include)
target_link_libraries(ctest bounds)
