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

#include "libheuristic.h"

#include "worker.h"
#include "fact_work.h"
#include "work.h"
#include "communicator.h"

worker::worker(pbab * _pbb) : pbb(_pbb),size(pbb->size)
{
    dwrk = std::make_shared<work>();

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);

    pthread_mutex_init(&mutex_wunit, &attr);
    pthread_mutex_init(&mutex_inst, &attr);
    pthread_mutex_init(&mutex_best, &attr);
    pthread_mutex_init(&mutex_end, &attr);
    pthread_mutex_init(&mutex_updateAvail, &attr);
    pthread_mutex_init(&mutex_trigger, &attr);

    pthread_mutex_init(&mutex_solutions, &attr);
    sol_ind_begin=0;
    sol_ind_end=0;
    max_sol_ind=2*arguments::heuristic_threads;
    solutions=(int*)malloc(max_sol_ind*size*sizeof(int));

    pthread_cond_init(&cond_updateApplied, NULL);
    pthread_cond_init(&cond_trigger, NULL);

    local_sol=new solution(pbb->size);
    for(int i=0;i<size;i++)local_sol->perm[i]=pbb->sltn->perm[i];

    reset();
}

worker::~worker()
{
    pthread_mutex_destroy(&mutex_wunit);
    pthread_mutex_destroy(&mutex_inst);
    pthread_mutex_destroy(&mutex_best);
    pthread_mutex_destroy(&mutex_end);
    pthread_mutex_destroy(&mutex_updateAvail);
    pthread_mutex_destroy(&mutex_trigger);

    pthread_mutex_destroy(&mutex_solutions);

    free(solutions);
}

void
worker::reset()
{
    end     = false;
    shareWithMaster = true;
    updateAvailable = false;

    pbb->stats.totDecomposed = 0;
    pbb->stats.johnsonBounds = 0;
    pbb->stats.simpleBounds  = 0;
    pbb->stats.leaves = 0;

    setNewBest(false);
}

void
worker::wait_for_trigger(bool& check, bool& best)
{
    // wait for an event to trigger communication
    pthread_mutex_lock_check(&mutex_trigger);
    while (!sendRequest && !newBest) {
        pthread_cond_wait(&cond_trigger, &mutex_trigger);
    }
    check = sendRequest;
    best  = newBest;
    pthread_mutex_unlock(&mutex_trigger);
}

void
worker::wait_for_update_complete()
{
    // printf("wait for upd \t");
    pthread_mutex_lock_check(&mutex_updateAvail);
    // signal update
    updateAvailable = true;
    // wait until done
    while (updateAvailable) {
        pthread_cond_wait(&cond_updateApplied, &mutex_updateAvail);
    }
    pthread_mutex_unlock(&mutex_updateAvail);
    // printf("... complete %d \n",comm->rank);
}

// use main thread....
void *
comm_thread(void * arg)
{
    worker * w = (worker *) arg;

    w->sendRequestReady = true;
    w->sendRequest      = false;
    w->setNewBest(false);

    int nbiter = 0;
    int dummy  = 11;

    solution* mastersol=new solution(w->pbb->size);

    int masterbest;

    int *msg_counter = new int[5];

    while (1) {
        //----------CHECK WORKER TERMINATION----------
        if (w->checkEnd()) break;

        //------------------WAIT ***** ------------------
        //...until triggered and get reason (checkpoint or best)
        bool doCheckpoint, doBest;
        w->wait_for_trigger(doCheckpoint, doBest);

        //---------CHECKPOINT : SEND intervals------------
        if (doCheckpoint) {
            //reset checkpoint-trigger
            pthread_mutex_lock_check(&w->mutex_trigger);
            w->sendRequest = false;
            pthread_mutex_unlock(&w->mutex_trigger);

            //convert to mpz integer intervals and sort...
            w->work_buf->fact2dec(w->dwrk);
            //...send to MASTER
            w->comm->send_work(w->dwrk, 0, WORK);
        } else if (doBest) {
        //-------------BEST : SEND local best-------------
            //reset best-trigger
            w->setNewBest(false);
            //get worker-best-solution and cost
            solution tmp(w->pbb->size);
            int tmpcost;

            w->pbb->sltn->getBestSolution(tmp.perm,tmpcost);
            tmp.cost.store(tmpcost);

            //send to master
            w->comm->send_sol(&tmp, 0, BEST);
        }

        nbiter++;

        // -----------------------------------------------
        // ---------------RECEIVE ANSWER------------------
        // -----------------------------------------------
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        msg_counter[status.MPI_TAG]++;

        switch(status.MPI_TAG)
        {
            case WORK: /* new work */
            {
                //the receive buffer...
                std::shared_ptr<work> rwrk(new work());

                FILE_LOG(logDEBUG4)<<"worker receives";
                w->comm->recv_work(rwrk, 0, MPI_ANY_TAG, &status);
                w->work_buf->dec2fact(rwrk);

                // wait unitl update applied
                w->wait_for_update_complete();
                break;
            }
            case BEST: /* improved best */
            {
                // printf("worker receive best\n");fflush(stdout);
                MPI_Recv(&masterbest, 1, MPI_INT, status.MPI_SOURCE, BEST, MPI_COMM_WORLD, &status);
                w->pbb->sltn->updateCost(masterbest);
                break;
            }
            case END: /* global termination*/
            {
                MPI_Recv(&dummy, 1, MPI_INT, 0, END, MPI_COMM_WORLD, &status);
                FILE_LOG(logINFO) << "Rank " << w->comm->rank << " terminates.";
                w->end = true;
                break;
            }
            case NIL: /*nothing : still receive master-best*/
            {
                w->comm->recv_sol(mastersol, 0, NIL, &status);
                // printf("receive NIL=== %d ===\n",w->pbb->sltn->cost );fflush(stdout);
                // MPI_Recv(&masterbest, 1, MPI_INT, 0, NIL, MPI_COMM_WORLD, &status);
                w->pbb->sltn->update(mastersol->perm,mastersol->cost);
                break;
            }
            case SLEEP:
            {
                MPI_Recv(&dummy, 1, MPI_INT, 0, SLEEP, MPI_COMM_WORLD, &status);
                usleep(10);
                w->shareWithMaster=false;
                break;
            }
            default:
            {
                FILE_LOG(logERROR) << "unknown message";
                exit(-1);
            }
        }

        // can handle new request now... set flag to true
        // work buffer can be reused!!!
        // if work units were received: the update was taken into account (see cond_updateApplied)
        // else: buffer was used only for send (operation completed)
        pthread_mutex_lock_check(&w->mutex_trigger);
        w->sendRequestReady = true;
        pthread_mutex_unlock(&w->mutex_trigger);
    }

    FILE_LOG(logINFO) << "----------Worker Message Count----------";
    FILE_LOG(logINFO) <<"WORK\t"<<msg_counter[WORK];
    FILE_LOG(logINFO) <<"BEST\t"<<msg_counter[BEST];
    FILE_LOG(logINFO) <<"NIL\t"<<msg_counter[NIL];
    FILE_LOG(logINFO) <<"END\t"<<msg_counter[END];
    FILE_LOG(logINFO) <<"SLEEP\t"<<msg_counter[SLEEP];
    FILE_LOG(logINFO) <<"---------------------------------------";

    delete[] msg_counter;

    // // confirm that termination signal received... comm thread will be joined
    // MPI_Send(&dummy, 1, MPI_INT, 0, END, MPI_COMM_WORLD);
    // printf("end-comm / iter:\t %d\n",nbiter);fflush(stdout);

    FILE_LOG(logDEBUG1) << "comm thread return";

    pthread_exit(0);
    // return NULL;
} // comm_thread

bool
worker::commIsReady()
{
    bool isReady = false;
    pthread_mutex_lock_check(&mutex_trigger);
    isReady = sendRequestReady;
    pthread_mutex_unlock(&mutex_trigger);
    return isReady;
}

void
worker::tryLaunchCommBest()
{
    if (commIsReady()) {
        pthread_mutex_lock_check(&mutex_trigger);
        sendRequestReady = false;
        newBest = true;
        pthread_cond_signal(&cond_trigger);
        pthread_mutex_unlock(&mutex_trigger);
        // pbb->sltn->newBest = false;
    }
}

void
worker::tryLaunchCommWork()
{
    if (commIsReady()) {
        pthread_mutex_lock_check(&mutex_wunit);
        getIntervals();// fill buffer (prepare SEND)
        pthread_mutex_unlock(&mutex_wunit);

        pthread_mutex_lock_check(&mutex_trigger);
        sendRequestReady = false;
        sendRequest      = true;// comm thread uses this to distinguish comm tasks  (best/checkpoint)
        pthread_cond_signal(&cond_trigger);
        pthread_mutex_unlock(&mutex_trigger);
        trigger = false;// reset...
    }
}

bool
worker::checkEnd()
{
    bool stop = false;
    pthread_mutex_lock_check(&mutex_end);
    stop = end;
    pthread_mutex_unlock(&mutex_end);
    return stop;
}

bool
worker::checkUpdate()
{
    bool doUpdate = false;
    pthread_mutex_lock_check(&mutex_updateAvail);
    doUpdate = updateAvailable;
    pthread_mutex_unlock(&mutex_updateAvail);
    return doUpdate;
}

bool worker::foundNewBest()
{
    bool ret;
    pthread_mutex_lock_check(&mutex_trigger);
    ret=newBest;
    pthread_mutex_unlock(&mutex_trigger);
    return ret;
}

void worker::setNewBest(bool _v){
    pthread_mutex_lock_check(&mutex_trigger);
    newBest=_v;
    pthread_mutex_unlock(&mutex_trigger);
};



// performs heuristic in parallel to exploration process
void *
heu_thread2(void * arg)
{
    pthread_detach(pthread_self());

    worker * w = (worker *) arg;

    // pthread_mutex_lock_check(&w->pbb->mutex_instance);
    // std::unique_ptr<fastNEH> neh = std::make_unique<fastNEH>(w->pbb->instance);
    std::unique_ptr<IG> ils = std::make_unique<IG>(w->pbb->instance.get());
    // IG ils(w->pbb->instance);
//     IG* ils=new IG(w->pbb->instance);
//     th->strategy=PRIOQ;
    // pthread_mutex_unlock(&w->pbb->mutex_instance);
//
    int N=w->pbb->size;
    subproblem *s=new subproblem(N);
//
    int gbest;
//     int cost;
//
    bool take=false;
//     std::cout<<"1 heuristic thread\n";
//
    while(!w->checkEnd()){
        w->pbb->sltn->getBestSolution(s->schedule.data(),gbest);// lock on pbb->sltn
//         // cost = gbest;
//
//         int c;
//         // w->local_sol->getBestSolution(s->schedule,c);
//
        int r=pbb::random::intRand(0,100);
//
//         take=false;
        pthread_mutex_lock_check(&w->mutex_solutions);
        if(w->sol_ind_begin < w->sol_ind_end && r<80){
            take=true;

            if(w->sol_ind_begin >= w->max_sol_ind){
                FILE_LOG(logERROR) << "Index out of bounds";
                exit(-1);
            }
            for(int i=0;i<N;i++){
                s->schedule[i]=w->solutions[w->sol_ind_begin*N+i];
            }

            // std::cout<<*s<<std::endl;
//
            w->sol_ind_begin++;
        }
        pthread_mutex_unlock(&w->mutex_solutions);

        s->limit1=-1;
        s->limit2=w->pbb->size;
//
//         // ils->perturbation(s->schedule, 3, 0, w->pbb->size);
//         // ils->igiter=100;
//         // ils->acceptanceParameter=1.6;
        int cost=ils->runIG(s);
//         // }
//         if(!take){
//             ils->perturbation(s->schedule, 3, 0, w->pbb->size);
//             ils->igiter=200;
//             ils->acceptanceParameter=1.5;
//             cost=ils->runIG(s);
//     		// subproblem reduced(w->pbb->size);
//             // int destroy=5;
//             // ils->destruction(s->schedule, reduced.schedule, destroy);
//     		// ils->localSearchPartial(s->schedule,w->pbb->size-destroy);
//             // ils->construction(s->schedule, reduced.schedule, destroy, 0, w->pbb->size);
//             // cost = ils->makespan(s);
//         }
//         // else{
//         //     ils->igiter=100;
//         //     // ils->acceptanceParameter=1.0;
//         //     cost=ils->runIG(s);
//         //     // cost = ils->makespan(s);
//         //     // ils->perturbation(s->schedule, 2, 0, w->pbb->size);
//         //
//         // }
//
//         cost=th->ITS(s,ccc);
//         // printf("heueuhh %d %d %d\n",ccc,cost,w->pbb->sltn->getBest());
//
        if (cost<w->pbb->sltn->getBest()){
            w->pbb->sltn->update(s->schedule.data(),cost);
            w->tryLaunchCommBest();
        }
        if(cost<w->local_sol->cost){
            w->local_sol->update(s->schedule.data(),cost);
            FILE_LOG(logINFO)<<"LocalBest "<<cost<<"\t"<<*(w->local_sol);
        }
    }
//
//     delete ils;
//     delete th;
//
    pthread_exit(0);
//     // return NULL;
}


// worker main thread : spawn communicator
void
worker::run()
{
    //------------create communication thread------------
    pthread_t * comm_thd;
    pthread_attr_t attr;

    comm_thd = (pthread_t *) malloc(sizeof(pthread_t));
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(comm_thd, &attr, comm_thread, (void *) this);

    //-------------create heuristic threads-------------
    int nbHeuThds=arguments::heuristic_threads;
    pthread_t heur_thd[100];
    for(int i=0;i<nbHeuThds;i++)
    {
        pthread_create(&heur_thd[i], NULL, heu_thread2, (void *) this);
    }
    FILE_LOG(logDEBUG) << "Created " << nbHeuThds << " heuristic threads.";
    int workeriter = 0;

    // printf("RUN %d\n",M);fflush(stdout);

    // ==========================================
    // worker main-loop :
    // do work or try acquire new work unit until END signal received
    // bool allEnd = false;
    while (1) {
        workeriter++;

        // if comm thread has set END flag, exit
        if (checkEnd()) {
            FILE_LOG(logINFO) << "End detected";
            break;
        }

        // if UPDATE flag set (by comm thread), apply update and signal
        // (comm thread is waiting until update applied)
        if (checkUpdate()) {
            FILE_LOG(logDEBUG) << "Update work unit";
            updateWorkUnit();// read buffer (RECV)
        }


        // work is done here... explore intervals(s)
        //        pbb->ttm->on(pbb->ttm->workerExploretime);

        // printf("Rank : %d\n",comm->rank);
        bool allEnd = doWork();

        if(arguments::heuristic_threads)
            getSolutions();

        if(foundNewBest()){
            // std::cout<<"try send best :\t"<<pbb->sltn->cost<<std::endl;
            FILE_LOG(logDEBUG) << "Try launch best-communication";
            tryLaunchCommBest();
        }
        else if(shareWithMaster || allEnd){
            FILE_LOG(logDEBUG) << "Try launch work-communicaion";
            tryLaunchCommWork();
        }
    }

    // int err;
    int err = pthread_join(*comm_thd, NULL);
    if (err)
    {
        FILE_LOG(logDEBUG) << "Failed to join comm thread " << strerror(err);
    }

    pbb->ttm->logElapsed(pbb->ttm->workerExploretime, "Worker exploration time\t");

    // confirm that termination signal received... comm thread will be joined
    int dummy = 42;
    MPI_Send(&dummy, 1, MPI_INT, 0, END, MPI_COMM_WORLD);
    // printf("end-comm / iter:\t %d\n",comm->rank);fflush(stdout);

    pthread_attr_destroy(&attr);
    free(comm_thd);
} // worker::run
