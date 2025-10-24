#include "arguments.h"
// #include "pbab.h"
// #include "ttime.h"
// #include "log.h"
//
// #include "libbounds.h"
// #include "gpubb.h"
#include "gpuerrchk.h"
#include "fsp_int8_eval.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

class pbab;

void initializeBoundFSP(pbab * pbb)
{
    int nbJob_h,nbMachines_h;

    // get instance data
    (pbb->inst->data)->seekg(0);
    (pbb->inst->data)->clear();

    *(pbb->inst->data) >> nbJob_h;
    *(pbb->inst->data) >> nbMachines_h;
    //
    // somme_h = 0;
    // for (int i = 1; i < nbMachines_h; i++) somme_h += i;
    //
	// nbJobPairs_h = 0;
	// for (int i = 1; i < nbJob_h; i++) nbJobPairs_h += i;
    //
    // // init bound for GPU
    // allocate_host_bound_tmp();
    //
    // for (int i = 0; i < nbMachines_h; i++) {
    //     fillMachine();
    //
    //     for (int j = 0; j < nbJob_h; j++)
    //         *(pbb->inst->data) >> tempsJob_h[i * nbJob_h + j];
    //     fillLag();
    //     fillTabJohnson();
    //     fillMinTempsArrDep();
	// 	fillSumPT();
    // }
    //
    // copyH2Dconstant();
    // free_host_bound_tmp();
    //
    // // gpuErrchk(cudaMalloc((void **) &front_d, nbIVM * nbMachines_h * sizeof(int)));
    // // gpuErrchk(cudaMalloc((void **) &back_d, nbIVM * nbMachines_h * sizeof(int)));
    //
    //
    // bd = std::make_unique<gpu_fsp_bound>(nbJob_h,nbMachines_h,nbIVM);
}


int
main(int argc, char ** argv)
{
    arguments::parse_arguments(argc, argv);
    std::cout<<"=== solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

    pbab * pbb = new pbab(pbb_instance::make_inst(arguments::problem, arguments::inst_name));

#ifdef WITH_GPU
    int dev=1;

    //use device 0 by default
    gpuErrchk(cudaSetDevice(dev));
    gpuErrchk(cudaFree(0));

    //Device Properties, see https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
#endif




    return EXIT_SUCCESS;
} // main
