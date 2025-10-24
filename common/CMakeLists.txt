set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#building library libbounds
add_library(common)

target_sources(common PRIVATE src/arguments.cpp)
target_include_directories(common PUBLIC ${PROJECT_SOURCE_DIR}/common/include)
