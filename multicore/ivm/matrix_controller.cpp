/*
 * "global" part of multi-core b&b
 * - work stealing
 * - termination detection (local)
 * - best (in pbb->sltn)
 */
#include <sys/sysinfo.h>
#include <unistd.h>

#include <memory>

#include "../../common/include/arguments.h"
#include "../../common/include/pbab.h"
#include "../../common/include/solution.h"
#include "../../common/include/ttime.h"
#include "../../common/include/macros.h"
#include "../../common/include/log.h"

#include "bbthread.h"
#include "sequentialbb.h"
#include "matrix_controller.h"

void
matrix_controller::initFullInterval()
{
    for (int i = 0; i < size; i++) {
        pos[0][i] = 0;
        end[0][i]  = size - i - 1;
    }
    state[0]=1;

    ivm_bound<int>::first=true;

    FILE_LOG(logDEBUG) << "WS list ";
    for(auto i:victim_list)
    {
        FILE_LOG(logDEBUG) << i;
    }
    FILE_LOG(logINFO) << "MC initialized";
}

bool
matrix_controller::solvedAtRoot()
{
    return dynamic_cast<ivmthread*>(bbb[0])->ivmbb->solvedAtRoot();
}

//nbint := number received intervals
void
matrix_controller::initFromFac(const unsigned int nbint, const int * ids, int * _pos, int * _end)
{
    if (nbint > M) {
        printf("cannot handle more than %d intervals\n", M);
        exit(-1);
    }
    if (nbint == 0) {
        printf("nothing received\n");
        return;
    }

    printf("init from fac %d\n",nbint);fflush(stdout);

    updatedIntervals = 1;

    // for (int k = 0; k < M; k++) {
    //     pthread_mutex_lock_check(&bbb[k]->mutex_ivm);
    //     bbb[k]->clear();//zero IVM, intreval..
    //     pthread_mutex_unlock(&bbb[k]->mutex_ivm);
    // }

    for (unsigned int k = 0; k < nbint; k++) {
        pthread_mutex_lock_check(&bbb[k]->mutex_ivm);
        unsigned int id = ids[k];

        if (id >= M) {
            printf("ID > nbIVMs!");
            exit(-1);
        }

        FILE_LOG(logDEBUG) << "initFromFac " << id;

        victim_list.remove(id);
        victim_list.push_front(id);// put in front

        state[id]=1;

        // initialize at interval [begin,end]
        bbb[id]->setRoot(pbb->root_sltn->perm);

        for (int i = 0; i < size; i++) {
            pos[id][i] = _pos[k * size + i];
            end[id][i] = _end[k * size + i];
        }

        state[0]=1;
        // bbb[id]->initAtInterval(pos + k * size, end + k * size);
        pthread_mutex_unlock(&bbb[k]->mutex_ivm);
    }
}

void
matrix_controller::getIntervals(int * pos, int * end, int * ids, int &nb_intervals, const unsigned int  max_intervals)
{
    memset(pos, 0, max_intervals * size * sizeof(int));
    memset(end, 0, max_intervals * size * sizeof(int));

    if (max_intervals < M) {
        FILE_LOG(logERROR)<<"MC:buffer too small";
        exit(-1);
    }

    int nbActive = 0;
    for (unsigned int k = 0; k < M; k++) {
        pthread_mutex_lock_check(&bbb[k]->mutex_ivm);//don't need it...
        if (!bbb[k]->isEmpty()) {
        // if (dynamic_cast<ivmthread*>(bbb[k])->ivmbb->IVM->beforeEnd()) {
            ids[nbActive] = k;
            dynamic_cast<ivmthread*>(bbb[k])->ivmbb->IVM->getInterval(&pos[nbActive * size],&end[nbActive * size]);

            // memcpy(&pos[nbActive * pbb->size],dynamic_cast<ivmthread*>(bbb[k])->ivmbb->IVM->posVect, pbb->size*sizeof(int));
            // memcpy(&end[nbActive * pbb->size],dynamic_cast<ivmthread*>(bbb[k])->ivmbb->IVM->endVect, pbb->size*sizeof(int));
            nbActive++;
        }
        pthread_mutex_unlock(&bbb[k]->mutex_ivm);
    }

    std::cout<<"got "<<nbActive<<" intervals.\n";




    nb_intervals = nbActive;
}

int
matrix_controller::getNbIVM()
{
    return M;
}



int
matrix_controller::work_share(unsigned id, unsigned thief)
{
    if(id==thief){perror("can't share with myself (mc)\n"); exit(-1);}
    if(id > M){
        perror("invalid victim ID (mc)\n"); exit(-1);
    }
    if(thief > M){
        perror("invalid thief ID (mc)\n"); exit(-1);
    }

    int ret = dynamic_cast<ivmthread*>(bbb[id])->shareWork(1, 2, dynamic_cast<ivmthread*>(bbb[thief])->ivmbb);// ->IVM);//, matrices[thief]->state, matrices[thief]->bestcost);

    return ret;
}


// void
// matrix_controller::unlockWaiting(int id)
// {
//     FILE_LOG(logDEBUG1) << "Unlock all waiting";
//
//     // unlock threads waiting for work
//     for (int i = 0; i < M; i++) {
//         if (i == id) continue;
//         pthread_mutex_lock_check(&bbb[i]->mutex_shared);
//         bbb[i]->receivedWork=true;
//         pthread_cond_signal(&bbb[i]->cond_shared);
//         pthread_mutex_unlock(&bbb[i]->mutex_shared);
//         // sem_post(&bbb[i]->sema_wait);
//     }
// }

// void
// matrix_controller::stop(int id)
// {
//     unlockWaiting(id);
//
//     int ret=pthread_barrier_wait(&barrier);
//     if(ret==PTHREAD_BARRIER_SERIAL_THREAD){
//         std::cout<<"STOP "<<id<<"\n";
//         FILE_LOG(logDEBUG1) << "=== stop "<<M<<" ===";
//     }
//     // sem_destroy(&bbb[id]->sema_wait);
// }

void
matrix_controller::interruptExploration()
{
    allEnd.store(true);
}

// run by multiple threads!!!
void
matrix_controller::explore_multicore()
{
    // get unique ID
    int id = explorer_get_new_id();

    FILE_LOG(logDEBUG) << id << " === got ID";

    if(!bbb[id]){
        bbb[id] = make_bbexplorer(id);
        bbb[id]->setRoot(root.data());
        FILE_LOG(logDEBUG) << id << " === allocated";
    }

    FILE_LOG(logDEBUG) << id << " === state " << state[id];

    if(state[id]==1){
        FILE_LOG(logDEBUG) << id << " === state 1";

        if(updatedIntervals){
            pthread_mutex_lock(&mutex_steal_list);
            dynamic_cast<ivmthread*>(bbb[id])->ivmbb->initAtInterval(pos[id].data(), end[id].data());
            pthread_mutex_unlock(&mutex_steal_list);
        }

        bbb[id]->setWorkState(true);
        pthread_mutex_lock(&mutex_steal_list);
        victim_list.remove(id);
        victim_list.push_front(id);// put in front
        pthread_mutex_unlock(&mutex_steal_list);
    }else{
        FILE_LOG(logDEBUG) << id << " === state 0";
        dynamic_cast<ivmthread*>(bbb[id])->ivmbb->clear();
        bbb[id]->setWorkState(false);
    }

    // stick_this_thread_to_core(id);

    //reset thread local
    // bbb[id]->reset();
    bbb[id]->reset_requestQueue();

    FILE_LOG(logDEBUG1) << std::flush;
    dynamic_cast<ivmthread*>(bbb[id])->ivmbb->count_decomposed = 0;

    int ret = pthread_barrier_wait(&barrier);
    if(ret==PTHREAD_BARRIER_SERIAL_THREAD)
    {
        updatedIntervals = 0;
        FILE_LOG(logDEBUG) << "=== start "<<M<<" exploration threads ===";
        std::cout<<"counter:\t"<<end_counter<<std::endl;
    }
    ret = pthread_barrier_wait(&barrier);

    // std::cout<<"NNNN 0 "<<dynamic_cast<ivmthread*>(bbb[id])->ivmbb->count_decomposed<<" "<<pbb->stats.totDecomposed<<"\n";
    // FILE_LOG(logDEBUG1) << std::flush;
    // pthread_barrier_wait(&barrier);

    // bool continuer = false;

    int req=0;
    int rep=0;
    int iter=0;

    int bestCost=INT_MAX;

    while (1) {
        pbb->sltn->getBest(bestCost);
        dynamic_cast<ivmthread*>(bbb[id])->ivmbb->setBest(bestCost);//

        pthread_mutex_lock_check(&bbb[id]->mutex_ivm);

        bool continuer = bbb[id]->bbStep();

        pthread_mutex_unlock(&bbb[id]->mutex_ivm);

        if (allEnd.load(std::memory_order_relaxed)) {
            FILE_LOG(logDEBUG) << "=== ALL END";
            break;
        }else if (!continuer){ // && !allEnd.load(std::memory_order_relaxed)) {
            req++;
            request_work(id);
        }else{
            rep++;
            try_answer_request(id);
        }

        iter++;

        if(!arguments::singleNode)
        {
            bool passed=pbb->ttm->period_passed(WORKER_BALANCING);
            if(atom_nb_steals>M || passed)
            {
                break;
            }

            if(pbb->foundNewSolution){
                break;
            }
        }
    }

    int nnodes = dynamic_cast<ivmthread*>(bbb[id])->ivmbb->count_decomposed;

    pbb->stats.totDecomposed+=nnodes;

    allEnd.store(true);
    stop(id);
}

void *
mcbb_thread(void * _mc)
{
    matrix_controller * mc = (matrix_controller *) _mc;

    mc->explore_multicore();

    return NULL;
}

void
matrix_controller::resetExplorationState()
{
    //reset global variables
    end_counter.store(0);// termination counter
    allEnd.store(false);
    atom_nb_explorers.store(0);// id_generator
    atom_nb_steals.store(0);// termination counter
}

bool
matrix_controller::next()
{
    resetExplorationState();
    // foundNew.store(0);

	FILE_LOG(logDEBUG) << "start next " << end_counter;

    pthread_t threads[100];

    for (unsigned i = 0; i < M; i++)
        pthread_create(&threads[i], NULL, mcbb_thread, (void *) this);

    for (unsigned i = 0; i < M; i++)
    {
        // int err = pthread_join(threadId, &ptr);
        int err = pthread_join(threads[i], NULL);
        if (err)
        {
            std::cout << "Failed to join Thread : " << strerror(err) << std::endl;
            return err;
        }
    }

    return allEnd.load();// || periodPassed);
}


int
matrix_controller::getSubproblem(int *ret, const int N)
{
    // printf("get\n");

    // gpuErrchk(cudaMemcpy(depth_histo_h,depth_histo_d,size*sizeof(int int),cudaMemcpyDeviceToHost));

    int count=0;

    for (unsigned i = 0; i < M; i++){
        if( !bbb[i]->isEmpty() ){
        // if( dynamic_cast<ivmthread*>(bbb[i])->ivmbb->IVM->beforeEnd() ){
            count++;
        }
    }

    if(count==0)return 0;


    int nb=std::min(count,N);
    count=0;

    for(unsigned i=0;i<M;i++)
    {
        if(count>=nb)break;

        if(!bbb[i]->isEmpty())
        // if(dynamic_cast<ivmthread*>(bbb[i])->ivmbb->IVM->beforeEnd())
        {
            for(int k=0;k<size;k++){
                dynamic_cast<ivmthread*>(bbb[i])->getSchedule(&ret[count*size]);
                // ret[count*size+k]=bbb[i]->bd->node->schedule[k];
            }
            count++;
        }
    }

    // printf(";;; %d %d %d\n",count,N,nb);
    return nb;
}
