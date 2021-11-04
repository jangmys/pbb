set(MyCppSources
    localsearch/fastneighborhood.cpp
    localsearch/ls.cpp
    neh/fastinsertremove.cpp
    neh/fastNEH.cpp
    ils/ils.cpp
    beam/beam.cpp
    beam/tree.cpp
    beam/treeheuristic.cpp
)

# target_include_directories(fspHeuristic PUBLIC ./include)
target_sources(fspHeuristic PRIVATE ${MyCppSources})

# message(STATUS ${MyCppSources})
