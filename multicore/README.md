# PBB@Multi-core

### Solving PFSP instances with PBB@Multicore

- Go to the `/build` folder.
- If the build succeeded, there is an executable `./multicore/bb`.

Several options should be configured for PBB@Multi-core:

1) to use the default configuration, pass the `-f` option
```
-f ../multicore/mcconfig.ini    
```
2) Some options can be configured via command-line arguments




If not configured otherwise, PBB@Multi-core will use all available CPU cores.

_________________________________________

##### Example 1 : Solve Taillard's instance Ta31 (50 jobs, 5 machines):

`./multicore/bb -f ../multicore/mcconfig.ini -z p=fsp,i=ta31`

- The output should be an optimal permutation of length 50 with Cmax=2724. You should see something like this
```
...
=================
Exploration stats
=================
TOT-BRANCHED:	7534
TOT-LEAVES:	4
Found optimal solution:
        Min Cmax:	2724
        Argmin Cmax:	2724	|	30 9 16 31 49 40 29 7 33 25 27 44 28 26 13 20 8 3 15 14 1 24 12 42 4 17 10 46 45 5 0 6 37 39 2 21 41 23 38 32 48 22 11 43 19 47 36 34 18 35
Walltime	:	 0.003165362
```
- Multiple runs will probably yield different optimal solutions (no uniqueness!). If you want ALL optimal solutions, set `findAll = true` in the `.ini` file. Caution : the algorithm will run MUCH longer!

##### Example 2 : Solve VFR instance VFR20_10_6 (20 jobs, 10 machines)

`./multicore/bb -f ../multicore/mcconfig.ini -z p=fsp,i=VFR20_10_6`

- The output should be an optimal permutation of length 20 with Cmax=1576
- It should take around one second, depending on your machines. The default configuration will use all available processing cores. If you want sequential execution, set `threads = 1` in the `.ini` file (-1 by default)

##### Example 3 : Generate and solve a random instance with 51 jobs and 7 machines

`./multicore/bb -f ../multicore/mcconfig.ini -z p=fsp,i=rand_51_7`

- The beginning of the output shows the generated processing times (in Unif(0,99)).
- A random 51x7 instance will usually be solved quickly...now try 51x10, 51x11, 51x12, 51x13, ... ("difficult" territory starts here...")


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

`./multicore/bb -f ../multicore/mcconfig.ini -z p=fsp,i=../evaluation/flowshop/data/file14_7.txt`

The provided instance name (`i=...`) is the relative path to the file. It must start with a `.`, so if the file is in the same directory as executable, the command becomes:

`./multicore/bb -f ../multicore/mcconfig.ini -z p=fsp,i=./file14_7.txt`
