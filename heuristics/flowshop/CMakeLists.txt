FILE(GLOB MyCppSources *.cpp
    localsearch/*.cpp
    neh/*.cpp
    ils/*.cpp
    beam/*.cpp
)

# target_include_directories(fspHeuristic PUBLIC ./include)
target_sources(fspHeuristic PRIVATE ${MyCppSources})

# message(STATUS ${MyCppSources})
