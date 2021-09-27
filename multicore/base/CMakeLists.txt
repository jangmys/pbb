FILE(GLOB MyCppSources *.cpp)

# target_sources(bb PRIVATE ${MyCppSources})
if (TARGET bb)
    target_sources(bb PUBLIC ${MyCppSources})
endif()
if (TARGET dbb)
    target_sources(dbb PUBLIC ${MyCppSources})
endif()
