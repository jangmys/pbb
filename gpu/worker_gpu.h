#ifndef WORKER_GPU_H
#define WORKER_GPU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include <atomic>
#include <memory>

#include "log.h"
#include "gpubb.h"

#include "communicator.h"
#include "worker.h"
#include "fact_work.h"

class pbab;
class communicator;
class matrix_controller;

class work;
class fact_work;
class solution;

class gpubb;

class worker_gpu : public worker {
    // private:
public:
    worker_gpu(pbab * _pbb) : worker(_pbb)
    {
        // M : how many intervals can be handled?
        #ifdef USE_GPU
        M    = arguments::nbivms_gpu;
        comm = std::make_unique<communicator>(M, pbb->size);
        cudaFree(0);

        int num_devices = 0;
        cudaGetDeviceCount(&num_devices);

        //mapping MPI_ranks to devices
        int device_nb = (comm->rank) % num_devices;
        cudaSetDevice(device_nb);

        int device,count;
        cudaGetDeviceCount(&count);
        cudaGetDevice(&device);
        FILE_LOG(logINFO) << comm->rank << " using device" << device <<"/" << count;

        gbb = new gpubb(pbb);
        gbb->initialize();// allocate IVM on host/device
        gbb->initializeBoundFSP();
        //	gbb->initializeBoundQAP();
        //	gbb->initializeBoundNQ();

        gbb->copyH2D();
        FILE_LOG(logDEBUG1) << "GPU Bound initialized";

        // for mpi calls...
        work_buf = std::make_shared<fact_work>(M, size);
        work_buf->max_intervals = M;
        work_buf->id = 0;
        #endif // ifdef USE_GPU
    }

    gpubb * gbb;

    void
    updateWorkUnit();
    bool
    doWork();
    void
    getIntervals();

    void
    getSolutions();

    void interrupt();

    // int M;
    // int size;
    // char type;

    // bool standalone;

    // communicator * comm;
    // pbab * pbb;

    // matrix_controller * mc;

    // std::shared_ptr<fact_work> work_buf;
    // std::shared_ptr<work> dwrk;

//    fact_work * work_buf;
    // solution * best_buf;

    // worker(pbab * _pbb);
    // ~worker();

    // int best;

    // void tryLaunchCommBest();
    // void tryLaunchCommWork();


    // pthread_barrier_t barrier;
    // pthread_mutex_t mutex_end;
    // pthread_mutex_t mutex_wunit;
    // pthread_mutex_t mutex_best;
    // pthread_mutex_t mutex_updateAvail;
    // pthread_cond_t cond_updateApplied;
    //
    // pthread_mutex_t mutex_trigger;
    // pthread_cond_t cond_trigger;
    //
    // volatile bool end;
    // volatile bool newBest;
    // volatile bool trigger;
    // volatile bool updateAvailable;
    // volatile bool sendRequest;
    // volatile bool sendRequestReady;
    //
    // void wait_for_trigger(bool& check, bool& best);
    // void wait_for_update_complete();
    // bool checkEnd();
    // bool checkUpdate();
    // bool commIsReady();
    //
    // void run();
    //
    // void test();
};

#endif // ifndef WORKER_H
