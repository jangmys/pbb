#include <sys/sysinfo.h>

#include <pthread.h>
#include <sched.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#include "macros.h"
#include "pbab.h"
#include "ttime.h"
#include "log.h"
#include "rand.hpp"

#include "libheuristic.h"

#include "worker.h"
#include "fact_work.h"
#include "work.h"
#include "communicator.h"

worker::worker(pbab * _pbb, unsigned int nbIVM, int _mpi_local_rank) : pbb(_pbb),size(pbb->size),M(nbIVM),mpi_local_rank(_mpi_local_rank),local_decomposed_count(0),comm(std::make_unique<communicator>(size)),
        work_buf(std::make_shared<fact_work>(M, size))
{
    std::cout<<"lrank "<<_mpi_local_rank<<"\n";

    dwrk = std::make_shared<work>();

    nb_heuristic_threads = arguments::heuristic_threads;

    pthread_barrier_init(&barrier, NULL, 2);// sync worker and helper thread

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
    max_sol_ind=2*nb_heuristic_threads;
    solutions=(int*)malloc(max_sol_ind*size*sizeof(int));

    pthread_cond_init(&cond_updateApplied, NULL);
    pthread_cond_init(&cond_trigger, NULL);

    reset();
}

worker::~worker()
{
    pthread_barrier_destroy(&barrier);

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
    updateAvailable = false;
    pbb->workUpdateAvailable.store(false,std::memory_order_seq_cst);

    pbb->stats.totDecomposed = 0;
    pbb->stats.johnsonBounds = 0;
    pbb->stats.simpleBounds  = 0;
    pbb->stats.leaves = 0;

    setNewBest(false);
}

//=====================================================
//functions for communication threads
//=====================================================
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
    pbb->workUpdateAvailable.store(true,std::memory_order_seq_cst);//break exploration loop

    // wait until done
    while (updateAvailable) {
        pthread_cond_wait(&cond_updateApplied, &mutex_updateAvail);
    }
    pthread_mutex_unlock(&mutex_updateAvail);
}

// dedicated communication thread : handles all communications with master
void *
comm_thread(void * arg)
{
    worker * w = (worker *) arg;

    w->sendRequestReady = true;
    w->sendRequest      = false;
    w->setNewBest(false);

    //synchronize with worker thread
    pthread_barrier_wait(&w->barrier);

    int nbiter = 0;
    int dummy  = 11;

    int *msg_counter = new int[10];

    while (1) {
        //----------CHECK WORKER TERMINATION----------
        if (w->checkEnd()) break;

        //------------------WAIT ***** ------------------
        //...until triggered (sendRequest set to true) and get reason (checkpoint or best)
        bool doCheckpoint, doBest;
        w->wait_for_trigger(doCheckpoint, doBest);

        if (doCheckpoint) {
            //---------CHECKPOINT : SEND intervals------------
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
            //send to master
            w->comm->send_sol(w->pbb->best_found.perm.data(), w->pbb->best_found.cost, 0, BEST);
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
            case WORK: /*modified work*/
            case NEWWORK: /* new work */
            {
                //the receive buffer (decimal intervals)...
                auto rwrk = std::make_shared<work>();
                //receive
                w->comm->recv_work(rwrk, 0, status.MPI_TAG, &status);
                //convert to factoradic
                // w->work_buf = convert::dec2fact(rwrk,w->pbb->size);
                w->work_buf->dec2fact(rwrk);
                //signal and wait
                w->wait_for_update_complete();
                break;
            }
            case BEST: /* improved best */
            case NIL: /*nothing : receive master-best anyway*/
            {
                // printf("worker receive best\n");fflush(stdout);
                int masterbest;
                MPI_Recv(&masterbest, 1, MPI_INT, 0, status.MPI_TAG, MPI_COMM_WORLD, &status);
                w->pbb->best_found.updateCost(masterbest);
                break;
            }
            case END: /* global termination*/
            {
                MPI_Recv(&dummy, 1, MPI_INT, 0, END, MPI_COMM_WORLD, &status);
                FILE_LOG(logINFO) << "Rank " << w->comm->rank << " terminates.";
                pthread_mutex_lock_check(&w->mutex_end);
                w->end = true;
                pthread_mutex_unlock(&w->mutex_end);
                break;
            }
            case SLEEP:
            {
                //master has no free work units and steal fails...retry, but wait a little (1 millisec)
                MPI_Recv(&dummy, 1, MPI_INT, 0, SLEEP, MPI_COMM_WORLD, &status);
                usleep(1000);
                break;
            }
            default:
            {
                std::cout<<"Fatal error : worker received message with unknown tag.\n";
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
    FILE_LOG(logINFO) <<"CommIterations\t"<<nbiter;
    FILE_LOG(logINFO) <<"Nodes decomposed\t"<<w->local_decomposed_count;

    FILE_LOG(logINFO) <<"WORK\t"<<msg_counter[WORK];
    FILE_LOG(logINFO) <<"NEWWORK\t"<<msg_counter[NEWWORK];
    FILE_LOG(logINFO) <<"BEST\t"<<msg_counter[BEST];
    FILE_LOG(logINFO) <<"NIL\t"<<msg_counter[NIL];
    FILE_LOG(logINFO) <<"END\t"<<msg_counter[END];
    FILE_LOG(logINFO) <<"SLEEP\t"<<msg_counter[SLEEP];
    FILE_LOG(logINFO) <<"---------------------------------------";

    delete[] msg_counter;

    FILE_LOG(logDEBUG1) << "comm thread return";

    pthread_exit(0);
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
        FILE_LOG(logINFO) <<"trigger comm best";

        pthread_mutex_lock_check(&mutex_trigger);
        sendRequestReady = false;
        newBest = true;
        pthread_cond_signal(&cond_trigger);
        pthread_mutex_unlock(&mutex_trigger);
    }
}

void
worker::tryLaunchCommWork()
{
    if (commIsReady()) {
        FILE_LOG(logINFO) <<"trigger comm work";

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
    std::unique_ptr<IG> ils = std::make_unique<IG>(*(w->pbb->inst.get()));
    // pthread_mutex_unlock(&w->pbb->mutex_instance);

    int N=w->pbb->size;
    std::shared_ptr<subproblem> s = std::make_shared<subproblem>(N);

    int gbest;

        // solutions=(int*)malloc(max_sol_ind*size*sizeof(int));

    while(!w->checkEnd()){
        w->pbb->best_found.getBestSolution(s->schedule.data(),gbest);// lock on pbb->best_found
        int r=intRand(0,100);

        pthread_mutex_lock_check(&w->mutex_solutions);
        if(w->sol_ind_begin < w->sol_ind_end && r<80){
            if(w->sol_ind_begin >= w->max_sol_ind){
                // FILE_LOG(logERROR) << "Index out of bounds";
                exit(-1);
            }
            for(int i=0;i<N;i++){
                s->schedule[i]=w->solutions[w->sol_ind_begin*N+i];
            }
            w->sol_ind_begin++;
        }
        pthread_mutex_unlock(&w->mutex_solutions);

        s->limit1=-1;
        s->limit2=w->pbb->size;

        int cost=ils->runIG(s,ils->igiter);
//
        if (cost<w->pbb->best_found.getBest()){
            w->pbb->best_found.update(s->schedule.data(),cost);
            w->tryLaunchCommBest();
        }
        if(cost<w->pbb->best_found.cost){
            w->pbb->best_found.update(s->schedule.data(),cost);
            // FILE_LOG(logINFO)<<"LocalBest "<<cost<<"\t"<<w->pbb->best_found;
        }
    }
    pthread_exit(0);
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
    pthread_t heur_thd[100];
    for(size_t i=0;i<nb_heuristic_threads;i++)
    {
        pthread_create(&heur_thd[i], NULL, heu_thread2, (void *) this);
    }
    FILE_LOG(logDEBUG) << "Created " << nb_heuristic_threads << " heuristic threads.";

    pthread_barrier_wait(&barrier);// synchronize with communication thread

    int workeriter = 0;
    int count_updates = 0;

    // ==========================================
    // worker main-loop :
    // do work or try acquire new work unit until END signal received
    // bool allEnd = false;
    while (1) {
        workeriter++;

        // if comm thread has set END flag, exit
        if (checkEnd()) {
            // FILE_LOG(logINFO) << "Worker : End detected";
            break;
        }

        // if UPDATE flag set (by comm thread), apply update and signal
        // (comm thread is waiting until update applied)
        if (checkUpdate()) {
            count_updates++;
            updateWorkUnit();// read buffer (RECV) --- SYNC WITH COMM THREAD
        }


        // work is done here... explore intervals(s)
        if(!foundNewBest())
            (void)doWork();

        if(nb_heuristic_threads)
            getSolutions(solutions);

        if(foundNewBest()){
            // std::cout<<"try send best :\t"<<pbb->sltn->cost<<std::endl;
            FILE_LOG(logDEBUG) << "Try launch best-communication";
            tryLaunchCommBest();
        }
        // else if(allEnd){
        else{
            FILE_LOG(logDEBUG) << "Try launch work-communicaion";
            tryLaunchCommWork();
        }
    }

    // int err;
    int err = pthread_join(*comm_thd, NULL);
    if (err)
    {
        // FILE_LOG(logDEBUG) << "Failed to join comm thread " << strerror(err);
    }

    // FILE_LOG(logINFO) << "#updates: "<<count_updates;

    pbb->ttm->logElapsed(pbb->ttm->workerExploretime, "Worker exploration time\t");

    // confirm that termination signal received... comm thread will be joined
    int dummy = 42;
    MPI_Send(&dummy, 1, MPI_INT, 0, END, MPI_COMM_WORLD);
    // printf("end-comm / iter:\t %d\n",comm->rank);fflush(stdout);

    pthread_attr_destroy(&attr);
    free(comm_thd);
} // worker::run
