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
    //------------------PARAMETER PARSING-----------------
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

    //------------------SET UP LOGGING--------------------
    FILELog::ReportingLevel() = logERROR;
#ifndef NDEBUG
    FILELog::ReportingLevel() = logDEBUG;
#endif
    FILE* log_fd = fopen(arguments::logfile, "w" );
    Output2FILE::Stream() = log_fd;

    pbab * pbb = new pbab(        pbb_instance::make_inst(arguments::problem, arguments::inst_name));

    //------------------SET INSTANCE----------------------
    // pbb->set_instance(
    //     pbb_instance::make_instance(arguments::problem, arguments::inst_name)
    // );

    pbb->set_initial_solution();
    std::cout<<"initial solution "<<*(pbb->best_found);


	arguments::singleNode=true;
    cudaFree(0);

    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);


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
