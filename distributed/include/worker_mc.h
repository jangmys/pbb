#ifndef WORKER_MC_H
#define WORKER_MC_H

#include <atomic>
#include <memory>

class pbab;
class communicator;

class matrix_controller;

class work;
class fact_work;
class solution;

#include "thread_controller.h"
#include "matrix_controller.h"

#include "communicator.h"
#include "fact_work.h"
#include "worker.h"

class worker_mc : public worker
{
public:
    worker_mc(pbab * _pbb) :
        worker(_pbb),
        mc(std::make_unique<matrix_controller>(pbb))
    {
        M    = mc->get_num_threads();
        comm = std::make_unique<communicator>(M, size);

        work_buf = std::make_shared<fact_work>(M, size);
        work_buf->max_intervals = M;
        work_buf->id = 0;
    };

    std::unique_ptr<matrix_controller> mc;

    void interrupt();
    void updateWorkUnit();
    bool doWork();
    void getIntervals();
    void getSolutions();

};

#endif // ifndef WORKER_H
