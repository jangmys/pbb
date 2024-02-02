#!/bin/bash

rm *.o *.so

g++ -O3 -Wall -std=c++14 -c add.cpp -o add.o

#heuristics


#eval
gcc -O3 -Wall -fPIC -c ../evaluation/flowshop/src/c_bound_simple.c -o c_bound_simple.o -I../evaluation/flowshop/include -I../evaluation
gcc -O3 -Wall -fPIC -c ../evaluation/flowshop/src/c_bound_johnson.c -o c_bound_johnson.o -I../evaluation/flowshop/include -I../evaluation

g++ -O3 -Wall -std=c++14 -fPIC -c ../evaluation/flowshop/src/bound_fsp_weak.cpp -o bound_fsp_weak.o -I../evaluation/flowshop/include -I../evaluation
g++ -O3 -Wall -std=c++14 -fPIC -c ../evaluation/flowshop/src/bound_fsp_strong.cpp -o bound_fsp_strong.o -I../evaluation/flowshop/include -I../evaluation
g++ -O3 -Wall -std=c++14 -fPIC -c ../evaluation/dummy/src/bound_dummy.cpp -o bound_dummy.o -I../evaluation/dummy/include -I../evaluation

gcc -O3 -Wall -fPIC -c ../evaluation/flowshop/src/c_taillard.c -o c_taillard.o -I../evaluation/flowshop/include -I../evaluation

g++ -O3 -Wall -std=c++14 -fPIC -c ../evaluation/flowshop/src/instance_taillard.cpp -o instance_taillard.o -I../evaluation/flowshop/include -I../evaluation
g++ -O3 -Wall -std=c++14 -fPIC -c ../evaluation/dummy/src/instance_dummy.cpp -o instance_dummy.o -I../evaluation/dummy/include -I../evaluation


#common
gcc -O3 -Wall -fPIC -c ../common/src/arguments.cpp -o arguments.o -I../common/include
gcc -O3 -Wall -fPIC -c ../common/src/incumbent.cpp -o incumbent.o -I../common/include -I../evaluation
gcc -O3 -Wall -fPIC -c ../common/src/pbab.cpp -o pbab.o -I../common/include -I../evaluation -I../heuristics -I../multicore/operators

gcc -O3 -Wall -fPIC -c ../common/src/misc.c -o misc.o -I../common/include
g++ -O3 -Wall -std=c++14 -fPIC -c ../common/src/subproblem.cpp -o subproblem.o -I../common/include



g++ -O3 -Wall -std=c++14 -fPIC -c ../multicore/base/thread_controller.cpp -o thread_controller.o -I../multicore/operators -I../multicore/base -I../multicore/do -I../common/include -I../evaluation

g++ -O3 -Wall -std=c++14 -fPIC -c ../multicore/ivm/ivm.cpp -o ivm.o -I../evaluation -I../common/include
g++ -O3 -Wall -std=c++14 -fPIC -c ../multicore/ivm/intervalbb.cpp -o intervalbb.o -I../evaluation -I../common/include -I../multicore/operators -I../multicore/base
g++ -O3 -Wall -std=c++14 -fPIC -c ../multicore/ivm/matrix_controller.cpp -o matrix_controller.o -I../multicore/ivm -I../multicore/operators -I../multicore/base -I../multicore/do -I../common/include -I../evaluation

g++ -O3 -Wall -std=c++14 -fPIC -c ../multicore/do/make_ivm_algo.cpp -o make_ivm_algo.o -I../common/include -I../evaluation -I../multicore/operators -I../multicore/base



gcc -O3 -Wall -fPIC -fopenmp -c ../heuristics/flowshop/neh/fastinsertremove.cpp -o fastinsertremove.o -I../multicore/operators -I../evaluation/flowshop/include -I../evaluation -I../common/include -I../heuristics -I../multicore/do -I../multicore/operators
gcc -O3 -Wall -fPIC -fopenmp -c ../heuristics/flowshop/neh/fastNEH.cpp -o fastNEH.o -I../multicore/operators -I../evaluation/flowshop/include -I../evaluation -I../common/include -I../heuristics -I../multicore/do -I../multicore/operators

g++ -O3 -Wall -std=c++14 -fPIC $(python3 -m pybind11 --includes) -c test_add.cpp -o test_add.o -I../heuristics -I../evaluation -I../common/include -I../multicore/ivm -I../multicore/operators -I../multicore/base
g++ -O3 -Wall -shared -std=c++14 $(python3 -m pybind11 --includes) test_add.o fastNEH.o fastinsertremove.o add.o bound_fsp_weak.o bound_fsp_strong.o bound_dummy.o c_bound_simple.o c_bound_johnson.o instance_taillard.o instance_dummy.o c_taillard.o subproblem.o misc.o matrix_controller.o ivm.o intervalbb.o thread_controller.o arguments.o incumbent.o make_ivm_algo.o -o test_add$(python3-config --extension-suffix)

gcc -O3 -Wall -fPIC -fopenmp -c ../heuristics/flowshop/beam/beam.cpp -o beam.o -I../multicore/operators -I../evaluation/flowshop/include -I../evaluation -I../common/include -I../heuristics -I../multicore/do -I../multicore/operators
g++ -O3 -Wall -std=c++14 -fPIC $(python3 -m pybind11 --includes) -c fsp_heuristics.cpp -o fsp_heuristics.o -I../heuristics -I../evaluation -I../common/include -I../multicore/ivm -I../multicore/operators -I../multicore/base
g++ -O3 -Wall -shared -std=c++14 $(python3 -m pybind11 --includes) fsp_heuristics.o fastNEH.o fastinsertremove.o bound_fsp_weak.o bound_fsp_strong.o bound_dummy.o c_bound_simple.o c_bound_johnson.o instance_taillard.o instance_dummy.o c_taillard.o subproblem.o misc.o matrix_controller.o ivm.o intervalbb.o thread_controller.o arguments.o incumbent.o make_ivm_algo.o -o fsp_heuristics$(python3-config --extension-suffix)


python3 test.py

# python3 -c "import test_add; print(test_add.adder(2,3)); a=test_add.instance_taillard('ta20'); print(a.get_job_number(20))"
