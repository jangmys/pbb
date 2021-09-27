#include <sys/sysinfo.h>

#include <pthread.h>
#include <sched.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#include "macros.h"
#include "pbab.h"
#include "solution.h"
#include "ttime.h"
#include "log.h"

#include "worker_mc.h"
#include "fact_work.h"
#include "work.h"
#include "communicator.h"

#include "../../multicore/base/thread_controller.h"
#include "../../multicore/ivm/matrix_controller.h"



void
worker_mc::interrupt()
{
    mc->interruptExploration();
}

bool
worker_mc::doWork()
{
    pbb->ttm->on(pbb->ttm->workerExploretime);

    // printf("dowork///\n");
    // bool triggerComm = false;

    FILE_LOG(logDEBUG) << "=== dowork== " << comm->rank;
    // printf("donework== %d =\n",comm->rank);

    pbb->foundNewSolution.store(false);
    pthread_mutex_lock_check(&mutex_wunit);
    mc->next();
    pthread_mutex_unlock(&mutex_wunit);

    pbb->ttm->off(pbb->ttm->workerExploretime);

    setNewBest(pbb->foundNewSolution);
    // setNewBest(mc->foundNew);

    // std::cout<<"MCFOUNDNEW "<<mc->foundNew<<" "<<pbb->foundNewSolution<<std::endl;

    return true; //triggerComm;// comm condition met
}

void
worker_mc::updateWorkUnit()
{
    printf("update work unit\n");

    pthread_mutex_lock_check(&mutex_wunit);
    mc->initFromFac(
        work_buf->nb_intervals,
        work_buf->ids,
        work_buf->pos,
        work_buf->end
    );
    pthread_mutex_unlock(&mutex_wunit);

    pthread_mutex_lock_check(&mutex_updateAvail);
    updateAvailable = false;
    pthread_mutex_unlock(&mutex_updateAvail);
    pthread_cond_signal(&cond_updateApplied);
}

// // copies work units from GPU (resp. thread-private IVMs) to communicator-buffer
// // --> prepare SEND
void
worker_mc::getIntervals()
{
    mc->getIntervals(
        work_buf->pos,
        work_buf->end,
        work_buf->ids,
        work_buf->nb_intervals,
        work_buf->max_intervals
    );
    dwrk->exploredNodes      = pbb->stats.totDecomposed;
    dwrk->nbLeaves           = pbb->stats.leaves;
    pbb->stats.totDecomposed = 0;
    pbb->stats.leaves        = 0;

    // printf("%d get %d intervals\n",comm->rank,work_buf->nb_intervals); fflush(stdout);
}


void
worker_mc::getSolutions()
{
    // printf("%d %d\n",sol_ind_begin,sol_ind_end);
    pthread_mutex_lock_check(&mutex_solutions);

    if(sol_ind_begin >= sol_ind_end){
        int nb=mc->getSubproblem(solutions,max_sol_ind);
        if(nb>0){
            sol_ind_begin=0;
            sol_ind_end=nb;
        }

    }

    pthread_mutex_unlock(&mutex_solutions);
}
