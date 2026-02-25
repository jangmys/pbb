set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#building library libcommon
add_library(common)

target_sources(common PRIVATE src/arguments.cpp)
target_sources(common PRIVATE src/subproblem.cpp)

target_include_directories(common PUBLIC ${PROJECT_SOURCE_DIR}/common/include)
