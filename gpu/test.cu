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
    //use device 0 by default
    gpuErrchk(cudaSetDevice(1));
    gpuErrchk(cudaFree(0));
#endif

    return EXIT_SUCCESS;
} // main
