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

#include "../../common/include/pbab.h"
#include "../../common/include/solution.h"
#include "../../common/include/ttime.h"
#include "../../common/include/macros.h"
#include "../../common/include/log.h"

#include "bbthread.h"
#include "intervalbb.h"
#include "matrix_controller.h"

matrix_controller::matrix_controller(pbab* _pbb,int _nthreads) : thread_controller(_pbb,_nthreads){
    resetExplorationState();

    state = std::vector<int>(get_num_threads(),0);

    for(unsigned i=0;i<get_num_threads();i++){
        pos.emplace_back(std::vector<int>(_pbb->size,0));
        end.emplace_back(std::vector<int>(_pbb->size,0));
    }
};

std::shared_ptr<bbthread>
matrix_controller::make_bbexplorer(){
    //initialize local (sequential) BB ----> different options...!
    return std::make_shared<ivmthread>(
        pbb,
        make_interval_bb(pbb,arguments::boundMode)
    );
}

void
matrix_controller::initFullInterval()
{
    //all empty
    std::fill(std::begin(state),std::end(state),0);

    //explorer 0 gets complete interval [0,n![
    for (int i = 0; i < pbb->size; i++) {
        pos[0][i] = 0;
        end[0][i]  = pbb->size - i - 1;
    }
    state[0]=1;

    // Intervalbb<int>::first=true;
}

//nbint := number received intervals
void
matrix_controller::initFromFac(const unsigned int nbint, const int * ids, int * _pos, int * _end)
{
    updatedIntervals = 1;

    for (unsigned int k = 0; k < nbint; k++) {
        unsigned int id = ids[k];
        assert(id < get_num_threads());

        // victim_list.remove(id);
        // victim_list.push_front(id);// put in front
        state[id]=1;

        bbb[id]->setRoot(pbb->root_sltn->perm, -1, pbb->size);
        for (int i = 0; i < pbb->size; i++) {
            pos[id][i] = _pos[k * pbb->size + i];
            end[id][i] = _end[k * pbb->size + i];
        }
    }
}

int
matrix_controller::work_share(unsigned id, unsigned thief)
{
    assert(id != thief);
    assert(id < get_num_threads());
    assert(thief < get_num_threads());

    int ret = std::static_pointer_cast<ivmthread>(bbb[id])->shareWork(std::static_pointer_cast<ivmthread>(bbb[thief]));

    return (int)(ret>0);
}

// run by multiple threads!!!
// in distributed setting re-entry is possible
void
matrix_controller::explore_multicore()
{
    //---------------get unique ID---------------
    int id = explorer_get_new_id();
    FILE_LOG(logDEBUG) << "=== got ID " << id;

    //------check if explorer already exists------
    if(!bbb[id]){
        //make sequential bb-explorer
        bbb[id] = make_bbexplorer();

        //set level 0 subproblems
        bbb[id]->setRoot(pbb->root_sltn->perm, -1, pbb->size);
        FILE_LOG(logDEBUG) << "=== made explorer ("<<id<<")";
        FILE_LOG(logDEBUG) << *(pbb->root_sltn);
        updatedIntervals = 1;
        // state[id]=1;
    }else{
        FILE_LOG(logDEBUG) << "=== explorer ("<<id<<") is ready";
    }

    int bestCost=INT_MAX;

    //get global best UB
    pbb->sltn->getBest(bestCost);
    //set local UB
    bbb[id]->setLocalBest(bestCost);


    if(state[id]==1){
        //has non-empty interval ...
        FILE_LOG(logDEBUG) << "=== state 1 ("<<id<<")";

        if(updatedIntervals){
            std::static_pointer_cast<ivmthread>(bbb[id])->ivmbb->initAtInterval(pos[id], end[id]);
        }

        bbb[id]->set_work_state(true);
    }else{
        //has empty interval
        FILE_LOG(logDEBUG) << "=== state 0 ("<<id<<")";
        std::static_pointer_cast<ivmthread>(bbb[id])->ivmbb->clear();
        bbb[id]->set_work_state(false);
    }

    //reset counters and request queue
    bbb[id]->reset_request_queue();
    std::static_pointer_cast<ivmthread>(bbb[id])->ivmbb->reset_node_counter();

    //make sure all are initialized
    int ret = pthread_barrier_wait(&barrier);
    if(ret==PTHREAD_BARRIER_SERIAL_THREAD)
    {
        updatedIntervals = 0;
        FILE_LOG(logDEBUG) << "=== start "<<get_num_threads()<<" exploration threads ===";
    }



    while (1) {
        //get global best UB
        pbb->sltn->getBest(bestCost);
        //set local UB
        bbb[id]->setLocalBest(bestCost);

        if (allEnd.load()) {
            break;
        }else if (!bbb[id]->bbStep()){
            request_work(id);
        }else{
            try_answer_request(id);
        }

#ifdef WITH_MPI
        if(is_distributed())
        {
            if(pbb->workUpdateAvailable.load(std::memory_order_relaxed))
            {
                FILE_LOG(logINFO) << "=== BREAK (update avail)";
                break;
            }
            if(atom_nb_steals>1)
            {
                FILE_LOG(logINFO) << "=== BREAK (nb_steals "<<atom_nb_steals<<" )";
                break;
            }
            if(pbb->foundNewSolution){
                FILE_LOG(logINFO) << "=== BREAK (new sol)";
                break;
            }

            bool passed=pbb->ttm->period_passed(WORKER_BALANCING);
            if(passed)
            {
                FILE_LOG(logINFO) << "=== BREAK (time passed)";
                break;
            }
        }
#endif
    }

    allEnd.store(true);

    FILE_LOG(logDEBUG) << "=== Exit exploration loop";

    pbb->stats.totDecomposed += std::static_pointer_cast<ivmthread>(bbb[id])->ivmbb->get_decomposed_count();
    pbb->stats.leaves += std::static_pointer_cast<ivmthread>(bbb[id])->ivmbb->get_leaves_count();

    stop(id);
}

void *
mcbb_thread(void * _mc)
{
    matrix_controller * mc = (matrix_controller *) _mc;
    mc->explore_multicore();
    return NULL;
}

bool
matrix_controller::next()
{
    resetExplorationState();

    pthread_t threads[100];

    for (unsigned i = 0; i < get_num_threads(); i++)
        pthread_create(&threads[i], NULL, mcbb_thread, (void *) this);

    for (unsigned i = 0; i < get_num_threads(); i++)
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


//try to get N subproblems from mc-explorer
int
matrix_controller::getSubproblem(int *ret, const int N)
{
    int countActive=0;
    //how many active?
    for (unsigned i = 0; i < get_num_threads(); i++){
        if( !bbb[i]->isEmpty() ){
            countActive++;
        }
    }
    //no active => can't get anything
    if(countActive==0)return 0;

    //
    int nb=std::min(countActive,N);
    int countTake=0;

    for(unsigned i=0;i<get_num_threads();i++)
    {
        if(countTake>=nb)break;

        if(!bbb[i]->isEmpty())
        {
            memcpy(&ret[countTake*pbb->size],
                std::static_pointer_cast<ivmthread>(bbb[i])->getNode().schedule.data(),
                pbb->size*sizeof(int)
            );
            countTake++;
        }
    }

    return nb;
}
