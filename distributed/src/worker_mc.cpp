#include <sys/sysinfo.h>

#include <pthread.h>
#include <sched.h>
#include <assert.h>

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

    FILE_LOG(logDEBUG) << "=== dowork== " << comm->rank;

    pbb->foundNewSolution.store(false);
    pthread_mutex_lock_check(&mutex_wunit);
    mc->next();
    pthread_mutex_unlock(&mutex_wunit);

    pbb->ttm->off(pbb->ttm->workerExploretime);

    setNewBest(pbb->foundNewSolution);

    return true; //triggerComm;// comm condition met
}

void
worker_mc::updateWorkUnit()
{
    assert(work_buf->nb_intervals <= mc->get_num_threads());
    if (work_buf->nb_intervals == 0) {
        return;
    }

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

// // copies work units from thread-private IVMs to communicator-buffer
// // --> prepare SEND
void
worker_mc::getIntervals()
{
    memset(work_buf->pos, 0, work_buf->max_intervals * pbb->size * sizeof(int));
    memset(work_buf->end, 0, work_buf->max_intervals * pbb->size * sizeof(int));
    memset(work_buf->ids, 0, work_buf->max_intervals * sizeof(int));

    assert(work_buf->max_intervals >= mc->get_num_threads());

    int nbActive = 0;
    for (unsigned int k = 0; k < mc->get_num_threads(); k++) {
        if (!mc->get_bbthread(k)->isEmpty()) {
            work_buf->ids[nbActive] = k;
            std::static_pointer_cast<ivmthread>(mc->get_bbthread(k))->getInterval(&work_buf->pos[nbActive * size],&work_buf->end[nbActive * size]);
            nbActive++;
        }
    }
    work_buf->nb_intervals = nbActive;

    dwrk->exploredNodes      = pbb->stats.totDecomposed;
    dwrk->nbLeaves           = pbb->stats.leaves;
    pbb->stats.totDecomposed = 0;
    pbb->stats.leaves        = 0;
}


void
worker_mc::getSolutions()
{
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
