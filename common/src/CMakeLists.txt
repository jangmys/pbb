FILE(GLOB MyCppSources *.cpp)

if (TARGET multicore)
    target_sources(multicore PUBLIC ${MyCppSources})
endif()
