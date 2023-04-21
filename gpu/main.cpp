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

    //mandatory options for single-node GPU
    arguments::singleNode=true;
    arguments::worker_type='g';


    pbab * pbb = new pbab(        pbb_instance::make_inst(arguments::problem, arguments::inst_name));

    //------------------SET INSTANCE----------------------
    // pbb->set_instance(
    //     pbb_instance::make_instance(arguments::problem, arguments::inst_name)
    // );

    pbb->set_initial_solution();

    std::cout<<"\t#Problem:\t\t"<<arguments::problem<<" / Instance"<<arguments::inst_name<<"\n";
    std::cout<<"\t#ProblemSize:\t\t"<<pbb->size<<"\n"<<std::endl;

    std::cout<<"\t#Worker type:\t\t"<<arguments::worker_type<<std::endl;
    std::cout<<"\t#GPU workers:\t\t"<<arguments::nbivms_gpu<<std::endl;
    std::cout<<"\t#Bounding mode:\t\t"<<arguments::boundMode<<std::endl;
    std::cout<<"\t#Branching:\t\t"<<arguments::branchingMode<<std::endl;

    std::cout<<"\t#Initial solution\n"<<pbb->best_found;

    //use device 0 by default
    cudaSetDevice(0);
    cudaFree(0);

    //start timer
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    int device,count;
    cudaGetDeviceCount(&count);
    cudaGetDevice(&device);
    printf(" === Device %d/%d ==\n", device+1, count);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    gpubb* gbb = new gpubb(pbb);//%numDevices);
    gbb->initialize(0);// allocate IVM on host/device
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
