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

    pbab * pbb = new pbab();//, bound1, bound2);
    pbb->set_initial_solution();

	// strcpy(arguments::inifile,"./gpuconfig.ini");
	arguments::singleNode=true;

    FILELog::ReportingLevel() = logINFO;
    FILE* log_fd = fopen( "./logs/bblog.txt", "w" );
    Output2FILE::Stream() = log_fd;

    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    // pbb->buildInitialUB();

    // ###############################
    // ###### SINGLE NODE ######## (no MPI)
    // ###############################
    cudaFree(0);

    int device_nb = 0;
    cudaSetDevice(device_nb);

    int device,count;
    cudaGetDeviceCount(&count);
    cudaGetDevice(&device);
    printf("=== Device %d/%d ==\n", device, count-1);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    gpubb* gbb = new gpubb(pbb);//%numDevices);
    gbb->initialize();// allocate IVM on host/device
#ifdef FSP
    printf("=== FSP\n");fflush(stdout);
    gbb->initializeBoundFSP();
#endif
#ifdef TEST
    printf("\n=== TEST\n");fflush(stdout);
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
