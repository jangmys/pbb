FILE(GLOB MyCppSources *.cpp)

if (TARGET bb)
    target_sources(bb PUBLIC ${MyCppSources})
endif()
if (TARGET dbb)
    target_sources(dbb PUBLIC ${MyCppSources})
endif()
