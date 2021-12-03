/*
 * "global" part of multi-core b&b
 * - work stealing
 * - termination detection (local)
 * - best (in pbb->sltn)
 */
#include <sys/sysinfo.h>
#include <unistd.h>
#include <assert.h>

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

matrix_controller::matrix_controller(pbab* _pbb) : thread_controller(_pbb){
    resetExplorationState();

    size = _pbb->size;
    state = std::vector<int>(M,0);

    // root = std::vector<int>(_pbb->size,0);
    // for(int i=0;i<_pbb->size;i++){
    //     root[i]=pbb->root_sltn->perm[i];
    // }

    for(unsigned i=0;i<M;i++){
        pos.emplace_back(std::vector<int>(_pbb->size,0));
        end.emplace_back(std::vector<int>(_pbb->size,0));
    }
};

// matrix_controller::~matrix_controller()
// {
//     // pthread_mutex_destroy(&mutex_steal_list);
//     // pthread_mutex_destroy(&mutex_end);
// };

std::shared_ptr<bbthread>
matrix_controller::make_bbexplorer(){
    //initialize local (sequential) BB
    return std::make_shared<ivmthread>(pbb);
}

void
matrix_controller::initFullInterval()
{
    //all empty
    for(int i=0;i<M;i++){
        state[i]=0;
    }
    //explorer 0 gets complete interval [0,n![
    for (int i = 0; i < size; i++) {
        pos[0][i] = 0;
        end[0][i]  = size - i - 1;
    }
    state[0]=1;

    sequentialbb<int>::first=true;
}

bool
matrix_controller::solvedAtRoot()
{
    return std::dynamic_pointer_cast<ivmthread>(bbb[0])->ivmbb->solvedAtRoot();
}

//nbint := number received intervals
void
matrix_controller::initFromFac(const unsigned int nbint, const int * ids, int * _pos, int * _end)
{
    assert(nbint <= M);

    if (nbint == 0) {
        return;
    }
    updatedIntervals = 1;

    for (unsigned int k = 0; k < nbint; k++) {
        pthread_mutex_lock_check(&bbb[k]->mutex_ivm);
        unsigned int id = ids[k];

        assert(id < M);

        victim_list.remove(id);
        victim_list.push_front(id);// put in front
        state[id]=1;

        bbb[id]->setRoot(pbb->root_sltn->perm);
        for (int i = 0; i < size; i++) {
            pos[id][i] = _pos[k * size + i];
            end[id][i] = _end[k * size + i];
        }
        pthread_mutex_unlock(&bbb[k]->mutex_ivm);
    }
}

void
matrix_controller::getIntervals(int * pos, int * end, int * ids, int &nb_intervals, const unsigned int  max_intervals)
{
    memset(pos, 0, max_intervals * size * sizeof(int));
    memset(end, 0, max_intervals * size * sizeof(int));

    assert(max_intervals >= M);

    int nbActive = 0;
    for (unsigned int k = 0; k < M; k++) {
        pthread_mutex_lock_check(&bbb[k]->mutex_ivm);//don't need it...
        if (!bbb[k]->isEmpty()) {
            ids[nbActive] = k;
            std::dynamic_pointer_cast<ivmthread>(bbb[k])->ivmbb->IVM->getInterval(&pos[nbActive * size],&end[nbActive * size]);
            nbActive++;
        }
        pthread_mutex_unlock(&bbb[k]->mutex_ivm);
    }

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
    assert(id != thief);
    assert(id < M);
    assert(thief < M);

    int ret = std::dynamic_pointer_cast<ivmthread>(bbb[id])->shareWork(std::dynamic_pointer_cast<ivmthread>(bbb[thief])->ivmbb);

    return ret;
}

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
    FILE_LOG(logDEBUG) << " === got ID" << id;

    if(!bbb[id]){
        //make sequential bb-explorer
        bbb[id] = make_bbexplorer();
        //set root
        bbb[id]->setRoot(pbb->root_sltn->perm);
        FILE_LOG(logDEBUG) << id << " === allocated";
    }

    if(state[id]==1){
        //has non-empty interval ...
        FILE_LOG(logDEBUG) << id << " === state 1";

        if(updatedIntervals){
            std::dynamic_pointer_cast<ivmthread>(bbb[id])->ivmbb->initAtInterval(pos[id], end[id]);
        }

        bbb[id]->setWorkState(true);
        //only for "honest" ws strategy
        pthread_mutex_lock(&mutex_steal_list);
        victim_list.remove(id);
        victim_list.push_front(id);// put in front
        pthread_mutex_unlock(&mutex_steal_list);
    }else{
        //has empty interval
        FILE_LOG(logDEBUG) << id << " === state 0";
        std::dynamic_pointer_cast<ivmthread>(bbb[id])->ivmbb->clear();
        bbb[id]->setWorkState(false);
    }

    bbb[id]->reset_requestQueue();
    std::dynamic_pointer_cast<ivmthread>(bbb[id])->ivmbb->count_decomposed = 0;

    int ret = pthread_barrier_wait(&barrier);
    if(ret==PTHREAD_BARRIER_SERIAL_THREAD)
    {
        updatedIntervals = 0;
        FILE_LOG(logDEBUG) << "=== start "<<M<<" exploration threads ===";
    }
    ret = pthread_barrier_wait(&barrier);

    int bestCost=INT_MAX;

    while (1) {
        //get best UB
        pbb->sltn->getBest(bestCost);
        //set best UB for multi-core BB
        std::dynamic_pointer_cast<ivmthread>(bbb[id])->ivmbb->setBest(bestCost);

        bool continuer = bbb[id]->bbStep();

        if (allEnd.load(std::memory_order_relaxed)) {
            FILE_LOG(logDEBUG) << "=== ALL END";
            break;
        }else if (!continuer){
            request_work(id);
        }else{
            try_answer_request(id);
        }

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

    pbb->stats.totDecomposed += std::dynamic_pointer_cast<ivmthread>(bbb[id])->ivmbb->count_decomposed;
    pbb->stats.leaves += std::dynamic_pointer_cast<ivmthread>(bbb[id])->ivmbb->count_leaves;

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
    atom_nb_steals.store(0);//count work thefts
}

bool
matrix_controller::next()
{
    resetExplorationState();
	FILE_LOG(logDEBUG) << "start next " << end_counter;

    pthread_t threads[100];

    for (unsigned i = 0; i < M; i++)
        pthread_create(&threads[i], NULL, mcbb_thread, (void *) this);

    for (unsigned i = 0; i < M; i++)
    {
        int err = pthread_join(threads[i], NULL);
        if (err)
        {
            std::cout << "Failed to join Thread : " << strerror(err) << std::endl;
            return err;
        }
    }
    return allEnd.load();
}


int
matrix_controller::getSubproblem(int *ret, const int N)
{
    int count=0;

    for (unsigned i = 0; i < M; i++){
        if( !bbb[i]->isEmpty() ){
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
        {
            for(int k=0;k<size;k++){
                std::dynamic_pointer_cast<ivmthread>(bbb[i])->getSchedule(&ret[count*size]);
            }
            count++;
        }
    }

    return nb;
}
