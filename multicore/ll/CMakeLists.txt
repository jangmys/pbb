FILE(GLOB MyCppSources *.cpp)

if (TARGET bb)
    target_sources(bb PUBLIC ${MyCppSources})
endif()
