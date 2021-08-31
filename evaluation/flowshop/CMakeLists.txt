FILE(GLOB MyCppSources *.cpp src/*.cpp)

target_include_directories(bounds PUBLIC ./include)
target_sources(bounds PRIVATE ${MyCppSources})
