#ifndef WORKER_MC_H
#define WORKER_MC_H

#include <atomic>
#include <memory>

#include "thread_controller.h"
#include "victim_selector.h"
#include "matrix_controller.h"
#include "worker.h"

//wrap MC-BB to adapt it to the distributed setting
class worker_mc : public worker
{
public:
    worker_mc(pbab * _pbb,int _nthreads) :
        worker(_pbb,_nthreads),
        mc(std::make_unique<matrix_controller>(pbb,_nthreads,true))
    {
        mc->set_victim_select(std::make_unique<RandomVictimSelector>(M));
    };

    std::unique_ptr<matrix_controller> mc;

    void updateWorkUnit();
    bool doWork();

    fact_work get_intervals();
    void getIntervals();
    void getSolutions(int*);
};

#endif // ifndef WORKER_H
