# PBB@Cluster

For now, PBB@Cluster can either use GPU-based worker (if built with the `-DPU=true` option) or multi-core-workers, but not both.

##### Example 1 : Single-GPU Single-node

If your system has a single GPU you can run

`mpirun -np 2 ./distributed/dbb -f ../bbconfig.ini -z p=fsp,i=ta30`

to launch the master process + 1 GPU-worker.

You can also run

`mpirun -np 3 ./distributed/dbb -f ../bbconfig.ini -z p=fsp,i=ta30`

to use 2 GPU-workers, but both will compete for the same GPU.


##### Example 2 : multiple multi-GPU nodes

Let's say you have two compute nodes with 4 GPUs each. The file `nodefile` contains the names of the two nodes, e.g.
```
nodeA
nodeB
```

To launch two processes per node :
```
mpirun -hostfile nodefile -map-by ppr:2:node ./distributed/dbb -f ../bbconfig.ini -z p=fsp,i=ta21
```



#### Checkpointing
