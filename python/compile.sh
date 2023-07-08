#!/bin/bash

rm *.o *.so

g++ -O3 -Wall -std=c++14 -c add.cpp -o add.o

gcc -O3 -Wall -fPIC -c ../common/src/misc.c -o misc.o -I../common/include
g++ -O3 -Wall -std=c++14 -fPIC -c ../common/src/subproblem.cpp -o subproblem.o -I../common/include


gcc -O3 -Wall -fPIC -c ../evaluation/flowshop/src/c_bound_simple.c -o c_bound_simple.o -I../evaluation/flowshop/include -I../evaluation
g++ -O3 -Wall -std=c++14 -fPIC -c ../evaluation/flowshop/src/bound_fsp_weak.cpp -o bound_fsp_weak.o -I../evaluation/flowshop/include -I../evaluation

gcc -O3 -Wall -fPIC -c ../evaluation/flowshop/src/c_taillard.c -o c_taillard.o -I../evaluation/flowshop/include -I../evaluation

g++ -O3 -Wall -std=c++14 -fPIC -c ../evaluation/flowshop/src/instance_taillard.cpp -o instance_taillard.o -I../evaluation/flowshop/include -I../evaluation

g++ -O3 -Wall -std=c++14 -fPIC $(python3 -m pybind11 --includes) -c test_add.cpp -o test_add.o -I../evaluation -I../common/include




g++ -O3 -Wall -shared -std=c++14 $(python3 -m pybind11 --includes) test_add.o add.o bound_fsp_weak.o c_bound_simple.o instance_taillard.o c_taillard.o subproblem.o misc.o -o test_add$(python3-config --extension-suffix)

python3 test.py

# python3 -c "import test_add; print(test_add.adder(2,3)); a=test_add.instance_taillard('ta20'); print(a.get_job_number(20))"
