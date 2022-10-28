#ifndef WORKER_GPU_H
#define WORKER_GPU_H

#include <atomic>
#include <memory>
#include <cuda.h>

#include "log.h"
#include "gpubb.h"
#include "worker.h"


//wrap GPU-BB to adapt it to the distributed setting
class worker_gpu : public worker {
public:
    worker_gpu(pbab * _pbb, unsigned int _nbIVM) :
        worker(_pbb,_nbIVM),
        gbb(std::make_unique<gpubb>(pbb))
    {
        //-----------mapping MPI_ranks to devices-----------
        int num_devices = 0;
        cudaGetDeviceCount(&num_devices);
        cudaSetDevice((comm->rank) % num_devices);

        int device;
        cudaGetDevice(&device);
        FILE_LOG(logINFO) << comm->rank << " using device" << device <<"/" << num_devices;

        //------------GPU-BB------------------------
        gbb->initialize();// allocate IVM on host/device
        gbb->initializeBoundFSP();
        gbb->copyH2D();
        FILE_LOG(logDEBUG1) << "GPU Bound initialized";
    }

    std::unique_ptr<gpubb> gbb;

    void
    updateWorkUnit();
    bool
    doWork();
    void
    getIntervals();

    void
    getSolutions();
};

#endif // ifndef WORKER_GPU_H
