# Multi-core B&B for Permutation Flow-shop Scheduling Problem (PFSP)

### Quick Start

#### Compilation
1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`
_______________

#### Solving PFSP instances with PBB@Multi-multicore

- Go to the `/build/multicore/` folder.

- If the build succeeded, there is an executable called `bb`.

The easiest way to run PBB@Multi-core is to use the default `.ini` file `/multicore/mcconfig.ini`. It can be passed to `bb` using the option `-f <relative-path>`

Several PFSP instances are provided in the `/evaluation` folder:
- Taillard's benchmark
- VFR benchmark
- Random
- Custom file

Let's go! (assuming you're in `/build/multicore` now)

_______________

##### Example 1 : Solve Taillard's instance Ta31 (50 jobs, 5 machines):

`./bb -f ../../multicore/mcconfig.ini -z p=fsp,i=ta31`

- The output should be an optimal permutation of length 50 with Cmax=2724.
- Multiple runs will probably yield different optimal solutions (no uniqueness!). If you want ALL optimal solutions, set `findAll = true` in the `.ini` file - caution : the algorithm will run MUCH longer!

##### Example 2 : Solve VFR instance VFR20_10_6 (20 jobs, 10 machines)

`./multicore -f ../../multicore/mcconfig.ini -z p=fsp,i=VFR20_10_6`

- The output should be an optimal permutation of length 20 with Cmax=1576
- It should take around one second, depending on your machines. The default configuration will use all available processing cores. If you want sequential execution, set `threads = 1` in the `.ini` file (-1 by default)

##### Example 3 : Generate and solve a random instance with 51 jobs and 7 machines

`./multicore -f ../../multicore/mcconfig.ini -z p=fsp,i=rand_51_7`

- The beginning of the output shows the generated processing times (in Unif(0,99)).
- A random 51x7 instance will usually be solved quickly...now try 51x10, 51x11, 51x12, 51x13, ... ("difficult territory start about here")


##### Example 4 : Read instance data from a file and solve

In the `evaluation/flowshop/data` folder there is a file named `file14_7.txt` which contains the following:

```
14 7

90 87 56 19 51 67 20 73 48 38 22  9 94 89
73 99 17  0 99 70 31  4 62 97 86 99 93 15
59 99 90 95 65 55 97  1 34 25 13 25 84 50
67 60 53 37  9 22 61 41 49 59 35 23 66 45
57 34 96 42 56 75 94 27 37 57  1 25 88 40
36 66 49 86 39 83 14 21 56 13 29  8  7 31
32 65 35 98 18 97 24 55 17 85 11  7 63 78
```

The first row gives (#Jobs,#Machines) and the following gives the (#Jobs x #Machines) matrix of processing times.

Solve the associated PFSP with:

`./multicore -f ../../multicore/mcconfig.ini -z p=fsp,i=../../evaluation/flowshop/data/file14_7.txt`

The provided instance name (`i=...`) is the relative path to the file. It must start with a `.`, so if the file is in the same directory as executable, the command becomes:

`./multicore -f ../../multicore/mcconfig.ini -z p=fsp,i=./file14_7.txt`
