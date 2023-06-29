#!/bin/bash

rm *.o *.so

g++ -O3 -Wall -std=c++14 -c add.cpp -o add.o
g++ -O3 -Wall -std=c++14 -fPIC $(python3 -m pybind11 --includes) -c test_add.cpp -o test_add.o -I../evaluation
g++ -O3 -Wall -shared -std=c++14 $(python3 -m pybind11 --includes) test_add.o add.o -o test_add$(python3-config --extension-suffix)

python3 -c "import test_add; print(test_add.adder(2,3))"
