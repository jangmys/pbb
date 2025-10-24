#include "arguments.h"
// #include "pbab.h"
// #include "ttime.h"
// #include "log.h"
//
// #include "libbounds.h"
// #include "gpubb.h"
#include "gpuerrchk.h"

#include <cuda.h>
#include <cuda_runtime.h>

int
main(int argc, char ** argv)
{
    arguments::parse_arguments(argc, argv);
    std::cout<<"=== solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

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
