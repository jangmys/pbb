### Check the CMake version
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

### Define project
project("PyPBB"
    VERSION 1.0.0
    DESCRIPTION "Python bindings for PBB"
LANGUAGES C CXX)

add_subdirectory(extern/pybind11)

# set(ENV{pybind11_DIR} /home/jan/miniconda3/envs/py_parallel/lib/python3.9/site-packages/pybind11)
# set(ENV{pybind11_DIR} ../extern/pybind11)

find_package(Python COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG)

pybind11_add_module(pypbb pypbb.cpp)

target_sources(pypbb PRIVATE ../evaluation/flowshop/src/c_bound_simple.c)
target_sources(pypbb PRIVATE ../evaluation/flowshop/src/c_bound_johnson.c)
target_sources(pypbb PRIVATE ../evaluation/flowshop/src/c_taillard.c)

target_sources(pypbb PRIVATE ../evaluation/flowshop/src/bound_fsp_weak.cpp)
target_sources(pypbb PRIVATE ../evaluation/flowshop/src/bound_fsp_weak_idle.cpp)
target_sources(pypbb PRIVATE ../evaluation/flowshop/src/bound_fsp_strong.cpp)
target_sources(pypbb PRIVATE ../evaluation/flowshop/src/instance_taillard.cpp)
target_sources(pypbb PRIVATE ../evaluation/flowshop/src/instance_vrf.cpp)

target_sources(pypbb PRIVATE ../evaluation/dummy/src/instance_dummy.cpp)
target_sources(pypbb PRIVATE ../evaluation/dummy/src/bound_dummy.cpp)


FILE(GLOB CommonCppSources ../common/src/*.cpp)
FILE(GLOB CommonCSources ../common/src/*.c)

target_sources(pypbb PRIVATE ${CommonCppSources})
target_sources(pypbb PRIVATE ${CommonCSources})


# target_sources(pypbb PRIVATE ../common/src/subproblem.cpp)
# target_sources(pypbb PRIVATE ../common/src/incumbent.cpp)
# target_sources(pypbb PRIVATE ../common/src/misc.c)
# target_sources(pypbb PRIVATE ../common/src/rand.cpp)
# target_sources(pypbb PRIVATE ../common/src/arguments.cpp)
# target_sources(pypbb PRIVATE ../common/src/ttime.cpp)
# target_sources(pypbb PRIVATE ../common/src/pbab.cpp)

target_sources(pypbb PRIVATE ../multicore/base/thread_controller.cpp)
target_sources(pypbb PRIVATE ../multicore/base/victim_selector.cpp)


target_sources(pypbb PRIVATE ../multicore/ivm/ivm.cpp)
target_sources(pypbb PRIVATE ../multicore/ivm/intervalbb.cpp)
target_sources(pypbb PRIVATE ../multicore/ivm/matrix_controller.cpp)

target_sources(pypbb PRIVATE ../multicore/do/make_ivm_algo.cpp)



target_sources(pypbb PRIVATE ../heuristics/flowshop/neh/fastinsertremove.cpp)
target_sources(pypbb PRIVATE ../heuristics/flowshop/localsearch/fastneighborhood.cpp)

target_sources(pypbb PRIVATE ../heuristics/flowshop/localsearch/ls.cpp)
target_sources(pypbb PRIVATE ../heuristics/flowshop/neh/fastNEH.cpp)
target_sources(pypbb PRIVATE ../heuristics/flowshop/ils/ils.cpp)

target_sources(pypbb PRIVATE ../heuristics/flowshop/beam/tree.cpp)
target_sources(pypbb PRIVATE ../heuristics/flowshop/beam/beam.cpp)






# add_subdirectory(../evaluation/ ./build_bound)
target_include_directories(pypbb PUBLIC ../evaluation/dummy/include)
target_include_directories(pypbb PUBLIC ../evaluation/flowshop/include)
target_include_directories(pypbb PUBLIC ../evaluation/)
target_include_directories(pypbb PUBLIC ../heuristics/)
target_include_directories(pypbb PUBLIC ../common/include)
target_include_directories(pypbb PUBLIC ../common/include)

target_include_directories(pypbb PUBLIC ../multicore/operators)
target_include_directories(pypbb PUBLIC ../multicore/base)
target_include_directories(pypbb PUBLIC ../multicore/ivm)
target_include_directories(pypbb PUBLIC ../multicore/do)
# Python_add_library(pypbb pypbb.cpp)
# target_link_libraries(MyModule2 pybind11::headers)
# set_target_properties(MyModule2 PROPERTIES
#                                 INTERPROCEDURAL_OPTIMIZATION ON
#                                 CXX_VISIBILITY_PRESET ON
#                                 VISIBILITY_INLINES_HIDDEN ON)


# target_sources(pypbb PRIVATE
#     # ../evaluation/flowshop/src/c_bound_simple.c
#     # ../evaluation/flowshop/src/c_bound_johnson.c
#     # ../evaluation/flowshop/src/c_taillard.c
#     pypbb.cpp
# )
#
# # target_link_libraries(pypbb PUBLIC )
#
# # message(STATUS ${pybind11_INCLUDE_DIRS})
# # message(STATUS ${pybind11_INCLUDE_DIR})
#
#
# # target_include_directories(pypbb PUBLIC ${pybind11_INCLUDE_DIR})
# target_include_directories(pypbb PUBLIC pybind11::pybind11)
