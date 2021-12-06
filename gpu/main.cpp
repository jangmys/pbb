#include "arguments.h"
#include "pbab.h"
#include "solution.h"
#include "ttime.h"
#include "log.h"

#include "libbounds.h"
#include "gpubb.h"

#include <cuda.h>

int
main(int argc, char ** argv)
{
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

    /*SET UP LOGGING*/
    FILELog::ReportingLevel() = logINFO;
    FILE* log_fd = fopen(arguments::logfile, "w" );
    Output2FILE::Stream() = log_fd;

    pbab * pbb = new pbab();
    pbb->set_initial_solution();
    std::cout<<"initial solution "<<*(pbb->sltn);

	arguments::singleNode=true;

    FILELog::ReportingLevel() = logINFO;

    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    cudaFree(0);

    int device_nb = 0;
    cudaSetDevice(device_nb);

    int device,count;
    cudaGetDeviceCount(&count);
    cudaGetDevice(&device);
    printf(" === Device %d/%d ==\n", device+1, count);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    gpubb* gbb = new gpubb(pbb);//%numDevices);
    gbb->initialize();// allocate IVM on host/device
#ifdef FSP
    gbb->initializeBoundFSP();
#endif
#ifdef TEST
    gbb->initializeBoundTEST();
#endif
    gbb->copyH2D();
    gbb->initFullInterval();

    gbb->next();

    gbb->getStats();

	pbb->printStats();

    delete gbb;

    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("\nWalltime :\t %2.8f\n", (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9f);

    return EXIT_SUCCESS;
} // main
