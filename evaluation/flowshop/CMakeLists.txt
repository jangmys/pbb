FILE(GLOB MyCppSources src/*.cpp src/*.c)

target_include_directories(bounds PUBLIC ./include)
target_sources(bounds PRIVATE ${MyCppSources})
