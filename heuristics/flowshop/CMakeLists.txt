FILE(GLOB MyCppSources *.cpp localsearch/*.cpp)
FILE(GLOB MyCppSources *.cpp neh/*.cpp)
FILE(GLOB MyCppSources *.cpp ils/*.cpp)
FILE(GLOB MyCppSources *.cpp beam/*.cpp)

# target_include_directories(fspHeuristic PUBLIC ./include)
target_sources(fspHeuristic PRIVATE ${MyCppSources})

# message(STATUS ${MyCppSources})
