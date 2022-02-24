# PBB: Parallel B&B for the Permutation Flow-shop Scheduling Problem (PFSP)

B&B is an exact algorithm which solves combinatorial optimization problems by dynamically constructing and exploring a search tree that implicitly enumerates all possible solutions.

Due to the combinatorial explosion of the search space, this requires massively parallel processing for all but the smallest problem instances. PBBPerm provides parallel B&B algorithms for solving permutation-based optimization problems on multi-core processors, GPUs and large-scale GPU-accelerated HPC systems.

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
    - `-DMPI=true`
    - `-DGPU=true`
4. `make`

_________________

### Functionalities
see
- [PBB@multicore](./multicore/README.md)
- [PBB@GPU](./multicore/README.md)
- [PBB@Cluster](./distributed/README.md)

___________________________________________

### Instances
PFSP Benchmark instances for are included in the `evaluation/flowshop/data` folder.

- [Taillard's instances](http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html) (1993)
    - for each of the 120 instances, the file `instances.data` contains the best-known makespans according to [this](http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html).
    - the instance data generator code is in `src/instance_taillard.cpp`
        - for each instance it contains hardcoded timeseeds and the linear congruential generator as in [E.Taillard's paper](http://mistic.heig-vd.ch/taillard/articles.dir/Taillard1993EJOR.pdf)

- [Vallada-Ruiz-Framinan (VRF) instances](http://soa.iti.es/problem-instances) (2017)
    - for each of the 480 instances the file `instancesVRF.data` contains the number of jobs and best LB/UB as provided by the authors [here](http://soa.iti.es/problem-instances)
    - the folder `vrf_parameters` contains the processing times and `src/instance_vrf.cpp` the code for reading the instance files.

- User-Undefined
    - File `file14_7.txt` is a sample instance (14 jobs, 7 machines) that is readable by PBB. Users can provide their instance data though similar files.

___________________

### Configuration
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
- [time parameters]
