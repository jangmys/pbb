FILE(GLOB MyCppSources *.cpp)

if (TARGET bb)
    target_sources(bb PUBLIC ${MyCppSources})
endif()

# if (TARGET llbb)
#     target_sources(bb PUBLIC ${MyCppSources})
# endif()

if (TARGET dbb)
    target_sources(dbb PUBLIC ${MyCppSources})
endif()
if (TARGET gpubb)
    target_sources(gpubb PUBLIC ${MyCppSources})
endif()
# if (TARGET fspHeuristic)
#     target_sources(fspHeuristic PUBLIC ${MyCppSources})
# endif()

if (TARGET testHeuristic)
    target_sources(testHeuristic PUBLIC ${MyCppSources})
endif()
