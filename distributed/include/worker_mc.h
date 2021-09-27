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
        mc(new matrix_controller(pbb))
    {
        // mc = new matrix_controller(pbb);
        M    = mc->getNbIVM();

        // TODO make a setter fucntion;;
        comm = new communicator(M, size);
        printf("construct workermc %d %d\n",M,size);

        work_buf = std::make_shared<fact_work>(M, size);
        work_buf->max_intervals = M;
        work_buf->id = 0;
    };

    matrix_controller * mc;

    void interrupt();
    void updateWorkUnit();
    bool doWork();
    void getIntervals();
    void getSolutions();

};

#endif // ifndef WORKER_H
