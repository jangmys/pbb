# PBB: Parallel B&B for the Permutation Flow-shop Scheduling Problem (PFSP)

B&B is an exact algorithm which solves combinatorial optimization problems by dynamically constructing and exploring a search tree that implicitly enumerates all possible solutions.

Due to the combinatorial explosion of the search space, this requires massively parallel processing for all but the smallest problem instances. PBB provides parallel B&B algorithms for solving permutation-based optimization problems on multi-core processors, GPUs and large-scale GPU-accelerated HPC systems.

## Contents
- [Compilation](#compilation)
- [PBB@multicore](./multicore/README.md)
- [PBB@GPU](./multicore/README.md)
- [PBB@Cluster](./distributed/README.md)
- [PFSP instances](#instances)
- [Configuration options](#configuration)



### Compilation

#### Prerequisites
- All
    - [CMake](https://cmake.org/) >=3.13
    - C++ compiler (tested gcc 7.5)
- For PBB@Cluster
    - [GNU Multiple Precision Arithmetic Library](https://gmplib.org/) (tested GMP 6.1.2)
    - [OpenMPI](https://www.open-mpi.org/) (tested OpenMPI 2.1.1)
- For PBB@GPU
    - [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (tested CUDA 11.1)

#### Build
To build PBB we use CMake>=3.13.

You can build PBB without GMP, MPI and CUDA - but PBB will be limited to multi-core execution.
In the PBB root directory:

1. `mkdir build`
2. `cd build`
3. `cmake <options> ..` where `<options>` are
    - `-DMPI=true` #enable MPI
    - `-DGPU=true` #enable GPU
    - `-DCMAKE_BUILD_TYPE=Release/Debug` #(un)define NDEBUG
4. `make`

This will build the following executables (if enabled)
- `build/multicore/bb`
- `build/distributed/dbb`
- `build/gpu/gpubb`


### Running PBB

#### Problem instance

The `-z` multioption allows to select a PFSP problem instance

```
-z p=fsp,i=ta20,o
```

where
- `p=fsp` : the problem (only pfsp, for now)
- `i=ta20` : the problem instance, here Taillard's instance ta20
- `o` : set the initial upper bound to optimum (read from file)

PFSP Benchmark instances for are included in the `evaluation/flowshop/data` folder.

The available options for the problem instance are:

- `ta1`-`ta120` : [Taillard's instances](http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html) (1993)
    - for each of the 120 instances, the file `instances.data` contains the best-known makespans according to [this](http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html).
    - the instance data generator code is in `src/instance_taillard.cpp`
        - for each instance it contains hardcoded timeseeds and the linear congruential generator as in [E.Taillard's paper](http://mistic.heig-vd.ch/taillard/articles.dir/Taillard1993EJOR.pdf)

- `VFR{N}_{M}_{I}` : [Vallada-Ruiz-Framinan (VRF) instances](http://soa.iti.es/problem-instances) (2017)
    - where {N} : jobs, {M} : machines, {I} : instance number
    - for each of the 480 instances the file `instancesVRF.data` contains the number of jobs and best LB/UB as provided by the authors [here](http://soa.iti.es/problem-instances)
    - the folder `vrf_parameters` contains the processing times and `src/instance_vrf.cpp` the code for reading the instance files.

- `file14_7.txt` : User-defined
    - read file located in the `evaluation/flowshop/data` folder, for example `file14_7.txt` with the following format:

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


### Command-line options
- `-z p=fsp,i=<instance>,o`
    - problem, instance, initial-ub
    - `instance` either of form `ta35` or `VFR_15_10_4`
    - `o` is optional for initializing UB best-known solution (from file)
- `-t <nbthreads>`
    - number of threads(multi-core)
- `--bound <0,1,2>`
    - incremental bounding of child nodes
    - full bound for child nodes
    - mixed mode    
- `--branch <-3,-2,-1,1,2,3>`
    - branching mode
- `--findall`,`-a`
    - changes pruning s.th. nodes with lb==ub are NOT pruned
- `--primary-bound <s,j>`
    - simple or johnson bound




### Configuration files

Some options can be given to PBB as command line arguments. But there are too many, so there are the following `.ini` files
    - `bbconfig.ini`
    - `multicore/mcconfig.ini`
    - `gpu/gpuconfig.ini`


### Ini-file options

Some option can be given to PBB as command line arguments. But there are too many, so there are the following `.ini` files
- `bbconfig.ini`
- `multicore/mcconfig.ini`
- `gpu/gpuconfig.ini`

Any of those files (or .ini files with same layout) can be passed to PBB with the `-f <relative-path to .ini file>` option

The options are:
- [problem]
    - `problem = flowshop`
        - only option for now
    - instance = ta20

        it should be more convenient to pass this through the `-z p=fsp,i=ta20` option

- [initial]
    - control how initial upper bound is generated

- [bb]
    - single node or distributed?
    - option for sorting sibling nodes in DFS
    - bounding mode (LB1,LB2,combined)
    - branching strategy ()
    - LB2 variants
    - search all

- [verbose]
    - toggle print solutions to stdout

- [time parameters]
    - global checkpoint interval (sec)
    - local checkpoint interval (sec)
    - timeout for support heuristic (sec)

- [multicore only]
    - number of threads
    - workstealing strategy

- [gpu only]
    - nb explorers

- [truncate bb search]

- [heuristic]

- [distributed only]



#### Functionalities
see
- [PBB@multicore](./multicore/README.md)
- [PBB@GPU](./multicore/README.md)
- [PBB@Cluster](./distributed/README.md)
