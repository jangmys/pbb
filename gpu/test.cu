#include <memory>

#include "arguments.h"
// #include "pbab.h"
// #include "ttime.h"
// #include "log.h"
//
#include "instance_factory.h"

// #include "libbounds.h"
// #include "gpubb.h"
#include "gpuerrchk.h"
#include "fsp_int8_eval.cuh"

#include <cuda.h>
#include <cuda_runtime.h>


void initializeBoundFSP(std::shared_ptr<instance_abstract> inst)
{
    int nbJob_h,nbMachines_h;

    // get instance data
    (inst->data)->seekg(0);
    (inst->data)->clear();

    *(inst->data) >> nbJob_h;
    *(inst->data) >> nbMachines_h;
    //
    // somme_h = 0;
    // for (int i = 1; i < nbMachines_h; i++) somme_h += i;
    //
	// nbJobPairs_h = 0;
	// for (int i = 1; i < nbJob_h; i++) nbJobPairs_h += i;
    //
    // // init bound for GPU
    // allocate_host_bound_tmp();
    //pbb_instance
    // for (int i = 0; i < nbMachines_h; i++) {
    //     fillMachine();cmake .. -DGMX_THREAD_MPI=OFF -DGMX_MPI=ON -DBUILD_SHARED_LIBS=OFF -DGMXAPI=OFF -DGMX_INSTALL_NBLIB_API=OFF -DGMX_DOUBLE=ON -DGMX_FFT_LIBRARY=fftw3 -DFFTWF_LIBRARY=/share/libraries/fftw/3.3.10-openmpi.4.1.5/lib/ -DFFTWF_INCLUDE_DIR=/share/libraries/fftw/3.3.10-openmpi.4.1.5/include -DGMX_BLAS_USER=/share/libraries/openblas/0.3.28/gcc/11.3.1/lib/ -DGMX_LAPACK_USER=/share/libraries/openblas/0.3.28/gcc/11.3.1/lib/ -DGMX_CP2K=ON -DCP2K_DIR=/share/applications/cp2k/v2025.2/lib/local/psmp/

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

    std::shared_ptr<instance_abstract> inst=pbb_instance::make_inst(arguments::problem, arguments::inst_name);
    // pbab * pbb = new pbab(pbb_instance::make_inst(arguments::problem, arguments::inst_name));

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
