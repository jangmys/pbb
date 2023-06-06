#include <sys/sysinfo.h>

#include <pthread.h>
#include <sched.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "communicator.h"
#include "worker_gpu.h"


bool
worker_gpu::doWork()
{
    pbb->ttm->on(pbb->ttm->workerExploretime);
    bool allEnd = false;
    allEnd = gbb->next();
    pbb->ttm->off(pbb->ttm->workerExploretime);

    setNewBest(gbb->localFoundNew);

    return allEnd; //triggerComm;// comm condition met
}

void
worker_gpu::updateWorkUnit()
{
    pthread_mutex_lock_check(&mutex_wunit);
    gbb->initFromFac(
        work_buf->nb_intervals,
        work_buf->ids,
        work_buf->pos,
        work_buf->end
    );
    pthread_mutex_unlock(&mutex_wunit);

    pthread_mutex_lock_check(&mutex_updateAvail);
    updateAvailable = false;
    pbb->workUpdateAvailable = false;
    pthread_mutex_unlock(&mutex_updateAvail);
    pthread_cond_signal(&cond_updateApplied);
}

// copies work units from GPU (resp. thread-private IVMs) to communicator-buffer
// --> prepare SEND
void
worker_gpu::getIntervals()
{
    gbb->getIntervals(
        work_buf->pos,
        work_buf->end,
        work_buf->ids,
        work_buf->nb_intervals,
        work_buf->max_intervals
    );

    work_buf->nb_decomposed = pbb->stats.totDecomposed;
    work_buf->nb_leaves     = pbb->stats.leaves;
    pbb->stats.totDecomposed = 0;
    pbb->stats.leaves        = 0;
}

void
worker_gpu::getSolutions(int* _solutions)
{
    pthread_mutex_lock_check(&mutex_solutions);
    if(sol_ind_begin >= sol_ind_end){
        int nb=gbb->getDeepSubproblem(_solutions,max_sol_ind);

        if(nb){
            sol_ind_begin=0;
            sol_ind_end=nb;
        }
    }
    pthread_mutex_unlock(&mutex_solutions);
}
