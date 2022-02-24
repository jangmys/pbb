# PBB@GPUs

### Solving PFSP instances with PBB@GPU

- Go to the `/build` folder.
- If the build succeeded, there is an executable `./gpu/gpubb`.

Several options should be configured for PBB@GPU. The easiest is to use the default configuration file `/gpu/gpuconfig.ini`. PBB@GPU will use 16384 explorers.

_________________________________________

##### Example 1 : Solve Taillard's instance Ta30 (20 jobs, 20 machines)

`./gpu/gpubb -f ../gpu/gpuconfig.ini -z p=fsp,i=ta30`

- PBB@GPU should find an optimal solution with makespan `Cmax=2178`

##### Example 2 : Solve VFR instance VFR20_15_1 (20 jobs, 15 machines)

`./gpu/gpubb -f ../gpu/gpuconfig.ini -z p=fsp,i=VFR20_15_1`

- The optimal makespan is `1936`
