add_library(bounds)

add_subdirectory(${PROJECT_SOURCE_DIR}/evaluation/flowshop)
add_subdirectory(${PROJECT_SOURCE_DIR}/evaluation/dummy)
target_include_directories(bounds PUBLIC ${PROJECT_SOURCE_DIR}/evaluation)

# install(TARGETS bounds ${PROJECT_SOURCE_DIR}/lib/)
# install(TARGETS bounds LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib/ COMPONENT libraries)

add_executable(testBounds test/testBounds.cpp)
target_link_libraries(testBounds bounds)
