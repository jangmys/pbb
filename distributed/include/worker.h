#ifndef WORKER_H
#define WORKER_H

#include <atomic>
#include <memory>

#include "rand.hpp"
#include "macros.h"
#include "communicator.h"
#include "pbab.h"
#include "fact_work.h"


class worker {
public:
    pbab * pbb;
    int size;
    int M;
    int mpi_local_rank;
    unsigned nb_heuristic_threads;
    unsigned long long int local_decomposed_count;

    std::unique_ptr<communicator> comm;
    std::shared_ptr<fact_work> work_buf;
    std::shared_ptr<work> dwrk;

    //heuristics ...
    unsigned int sol_ind_begin;
    unsigned int sol_ind_end;
    unsigned int max_sol_ind;
    int *solutions;
    pthread_mutex_t mutex_solutions;

    worker(pbab * _pbb, unsigned int _nbIVM,int _mpi_local_rank = 0);
    virtual ~worker();

    void tryLaunchCommBest();
    void tryLaunchCommWork();

    virtual void getSolutions(int*) = 0;
    virtual void getIntervals() = 0;
    virtual void updateWorkUnit() = 0;
    virtual bool doWork() = 0;

    pthread_barrier_t barrier;
    // pthread_mutex_t mutex_inst;
    pthread_mutex_t mutex_end;//protects bool end
    pthread_mutex_t mutex_wunit;//get intervals, init from WU
    // pthread_mutex_t mutex_best;
    pthread_mutex_t mutex_updateAvail;//prtects bool updateAvailable
    pthread_cond_t cond_updateApplied;

    pthread_mutex_t mutex_trigger;
    pthread_cond_t cond_trigger;

    volatile bool end;
    volatile bool trigger;
    volatile bool updateAvailable;
    volatile bool sendRequest;
    volatile bool sendRequestReady;

    volatile bool newBest;
    bool foundNewBest();
    void setNewBest(bool _v);

    void wait_for_trigger(bool& check, bool& best);
    void wait_for_update_complete();
    bool checkEnd();
    bool checkUpdate();
    bool commIsReady();

    void reset();
    void run();
protected:
    /*
    using pthreads... void* thdroutine(void*)
    mixing C and C++... keep everything public.
    */
};

#endif // ifndef WORKER_H
