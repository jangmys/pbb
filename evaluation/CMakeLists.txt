add_library(bounds)

add_subdirectory(${PROJECT_SOURCE_DIR}/evaluation/flowshop)
target_include_directories(bounds PUBLIC ${PROJECT_SOURCE_DIR}/evaluation)

add_executable(testBounds test/testBounds.cpp)
target_link_libraries(testBounds bounds)
