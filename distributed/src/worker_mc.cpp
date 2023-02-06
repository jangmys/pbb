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
#include "ttime.h"
#include "log.h"

#include "worker_mc.h"
#include "fact_work.h"
#include "work.h"
#include "communicator.h"

#include "../../multicore/base/thread_controller.h"
#include "../../multicore/ivm/matrix_controller.h"


// void
// worker_mc::interrupt()
// {
//     mc->interruptExploration();
// }

bool
worker_mc::doWork()
{
    FILE_LOG(logDEBUG1) << " === worker_mc::doWork (rank " << comm->rank<<")";

    pbb->ttm->on(pbb->ttm->workerExploretime);
    pbb->best_found.foundNewSolution.store(false);
    mc->next();
    pbb->ttm->off(pbb->ttm->workerExploretime);

    FILE_LOG(logDEBUG1) << " === worker_mc::doWork return from next (rank " << comm->rank<<")";

    //redundant?
    setNewBest(pbb->best_found.foundNewSolution.load());

    return true; //triggerComm;// comm condition met
}

void
worker_mc::updateWorkUnit()
{
    //INTERSECT IF UPDATE (not new work)?
    FILE_LOG(logINFO) << "ID "<< dwrk->id << " "<<work_buf->id;
    //
    // if(dwrk->id == work_buf->id){
    //     pthread_mutex_lock_check(&mutex_wunit);
    //     auto oldsz = dwrk->wsize();
    //     FILE_LOG(logINFO) << "OLD "<<*dwrk;
    //
    //     auto tmpwrk = std::make_shared<work>();
    //     auto recvwrk = std::make_shared<work>();
    //
    //     auto tmp_fwrk = get_intervals();
        // tmp_fwrk.fact2dec(tmpwrk);
    //     FILE_LOG(logINFO) << "CURRENT "<<*tmpwrk;
    //
    //     // getIntervals();
    //     // work_buf->fact2dec(tmpwrk);
    //
    //     work_buf->fact2dec(recvwrk);
    //     FILE_LOG(logINFO) << "RECV "<<*recvwrk;
    //     // FILE_LOG(logINFO) << *recvwrk;
    //     auto newsz = tmpwrk->wsize();
    //     // FILE_LOG(logINFO) << "UPDATE\t"<< oldsz << "\t"<< newsz << " "<< oldsz-newsz;
    //
    //     recvwrk->intersection(tmpwrk);
    //     FILE_LOG(logINFO) << "INTERSECT "<<*recvwrk;
    //
    //     work_buf->dec2fact(recvwrk);
    //
    //     pthread_mutex_unlock(&mutex_wunit);
    // }

    // FILE_LOG(logDEBUG) << " === update work unit (rank " << comm->rank<<")";

    assert(work_buf->nb_intervals <= mc->get_num_threads());

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
    pbb->workUpdateAvailable.store(false);
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

    // std::cout<<"GET INTERVAL===============\n";
    int nbActive = 0;
    for (unsigned int k = 0; k < mc->get_num_threads(); k++) {
        if (mc->get_ivmbb(k)->get_ivm()->beforeEnd()) {
            work_buf->ids[nbActive] = k;
            mc->get_ivmbb(k)->get_ivm()->getInterval(&work_buf->pos[nbActive * size],&work_buf->end[nbActive * size]);
            nbActive++;
        }
    }
    work_buf->nb_intervals = nbActive;

    work_buf->nb_decomposed = pbb->stats.totDecomposed;
    work_buf->nb_leaves     = pbb->stats.leaves;

    // dwrk->nb_decomposed      = pbb->stats.totDecomposed;
    // dwrk->nb_leaves           = pbb->stats.leaves;
    pbb->stats.totDecomposed = 0;
    pbb->stats.leaves        = 0;
}




fact_work
worker_mc::get_intervals()
{
    fact_work tmp(M,size);

    memset(tmp.pos, 0, work_buf->max_intervals * pbb->size * sizeof(int));
    memset(tmp.end, 0, work_buf->max_intervals * pbb->size * sizeof(int));
    memset(tmp.ids, 0, work_buf->max_intervals * sizeof(int));

    assert(tmp.max_intervals >= mc->get_num_threads());

    // std::cout<<"GET INTERVAL===============\n";
    int nbActive = 0;
    for (unsigned int k = 0; k < mc->get_num_threads(); k++) {
        if (mc->get_ivmbb(k)->get_ivm()->beforeEnd()) {
            tmp.ids[nbActive] = k;
            mc->get_ivmbb(k)->get_ivm()->getInterval(&work_buf->pos[nbActive * size],&work_buf->end[nbActive * size]);
            nbActive++;
        }
    }
    tmp.nb_intervals = nbActive;

    dwrk->nb_decomposed      = pbb->stats.totDecomposed;
    dwrk->nb_leaves           = pbb->stats.leaves;
    pbb->stats.totDecomposed = 0;
    pbb->stats.leaves        = 0;

    return tmp;
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
