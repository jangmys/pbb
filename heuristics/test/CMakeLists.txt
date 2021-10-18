add_executable(testHeuristic testHeuristic.cpp)

target_link_libraries(testHeuristic fspHeuristic)

target_include_directories(testHeuristic PUBLIC ${PARENT_DIR}/heuristics)
#
# target_include_directories(testHeuristic PUBLIC ${PARENT_DIR}/evaluation)
# target_include_directories(testHeuristic PUBLIC ${PARENT_DIR}/heuristics)
target_include_directories(testHeuristic PUBLIC ${PARENT_DIR}/heuristics/flowshop/ils)
target_include_directories(testHeuristic PUBLIC ${PARENT_DIR}/heuristics/flowshop/localsearch)
