#include "arguments.h"
#include "pbab.h"
#include "ttime.h"
#include "log.h"

#include "libbounds.h"
#include "gpubb.h"

#include <cuda.h>

int
main(int argc, char ** argv)
{
#ifdef WITH_GPU
    //use device 0 by default
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaFree(0));
#endif

    return EXIT_SUCCESS;
} // main
