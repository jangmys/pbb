/*
 * "global" part of multi-core b&b
 * - work stealing
 * - termination detection (local)
 * - best (in pbb->best_found)
 */
#include <sys/sysinfo.h>
#include <unistd.h>
#include <assert.h>

#include <memory>

#include <pbab.h>
#include <ttime.h>
#include <macros.h>
#include <log.h>

#include <intervalbb.h>
#include <matrix_controller.h>

#include "make_ivm_algo.h"

matrix_controller::matrix_controller(pbab* _pbb,int _nthreads) : ThreadController(_pbb,_nthreads){
    ivmbb = std::vector< std::shared_ptr<Intervalbb<int>> >(get_num_threads(),nullptr);

    state = std::vector<int>(get_num_threads(),0);
    for(unsigned i=0;i<get_num_threads();i++){
        pos.emplace_back(std::vector<int>(_pbb->size,0));
        end.emplace_back(std::vector<int>(_pbb->size,0));
    }
    pthread_mutex_init(&mutex_buffer,NULL);
};

//nbint := number received intervals
void
matrix_controller::initFromFac(const unsigned int nbint, const int * ids, int * _pos, int * _end)
{
    FILE_LOG(logINFO) << "=== init from factorial ";

    updatedIntervals=1;

    //CLEAR ALL BEFORE REFILL
    for(unsigned i=0;i<get_num_threads();i++){
        state[i]=0;
        for (int j = 0; j < pbb->size; j++) {
            pos[i][j] = 0;
            end[i][j] = 0;
        }
        pos[i][0]=pbb->size;
    }

    for (unsigned int k = 0; k < nbint; k++) {
        unsigned int id = ids[k];
        assert(id < get_num_threads());

        state[id]=1;

        for (int i = 0; i < pbb->size; i++) {
            pos[id][i] = _pos[k * pbb->size + i];
            end[id][i] = _end[k * pbb->size + i];
        }
    }
}

int
matrix_controller::work_share(unsigned id, unsigned thief_id)
{
    assert(id != thief);
    assert(id < get_num_threads());
    assert(thief < get_num_threads());

    int numShared = 0;
    int l         = 0;

    std::shared_ptr<ivm> thief(ivmbb[thief_id]->get_ivm());
    std::shared_ptr<ivm> IVM(ivmbb[id]->get_ivm());

    while (IVM->getPosition(l) == IVM->getEnd(l) && l < IVM->getDepth() && l < pbb->size - 4) l++;

    if (IVM->getPosition(l) < IVM->getEnd(l))
    {
        numShared++;
        for (int i = 0; i < l; i++) {
            thief->setPosition(i, IVM->getPosition(i));
            thief->setRow(i,IVM->getRowPtr(i));
            thief->setDirection(i, IVM->getDirection(i));
        }
        for (int i = 0; i < pbb->size; i++) thief->setEnd(i, IVM->getEnd(i));

        thief->setRow(l,IVM->getRowPtr(l));
        thief->setDirection(l, IVM->getDirection(l));
        thief->setPosition(l,IVM->cuttingPosition(l, 2));
        IVM->setEnd(l, thief->getPosition(l) - 1);

        // remaining levels : align thief left, victim right
        for (int i = l + 1; i < pbb->size; i++) thief->setPosition(i, 0);
        for (int i = l + 1; i < pbb->size; i++) IVM->setEnd(i, pbb->size - i - 1);

        thief->setDepth(l);
    }

    return (int)(numShared>0);
}

int
stick_this_thread_to_core(int core_id)
{
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);

    if (core_id < 0 || core_id >= num_cores)
        return EINVAL;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

// run by multiple threads!!!
// in distributed setting re-entry is possible
void
matrix_controller::explore_multicore()
{
    //---------------get unique ID---------------
    int id = explorer_get_new_id();
    FILE_LOG(logDEBUG) << "=== got ID " << id;

    stick_this_thread_to_core(id);

    //------check if explorer already exists------
    if(!ivmbb[id]){
        //make sequential bb-explorer
        ivmbb[id] = make_ivmbb<int>(pbb);
        //thread-local data for MC exploration
        thd_data[id] = std::make_shared<RequestQueue>();

        //set level 0 subproblems
        ivmbb[id]->setRoot(pbb->best_found.initial_perm.data());
        updatedIntervals = 1;
        state[id]=1;
    }else{
        FILE_LOG(logDEBUG) << "=== explorer ("<<id<<") is ready";
    }

    (void)pthread_barrier_wait(&barrier);

    int bestCost=INT_MAX;

    //get global best UB
    pbb->best_found.getBest(bestCost);
    //set local UB
    ivmbb[id]->setBest(bestCost);

    if(updatedIntervals){
        // std::cout<<"ID "<<id<<" init at interval\n";
        pthread_mutex_lock_check(&mutex_buffer);
        bool _active = ivmbb[id]->initAtInterval(pos[id], end[id]);
        pthread_mutex_unlock(&mutex_buffer);

        thd_data[id]->has_work.store(_active);
    }

    //reset counters and request queue
    thd_data[id]->reset_request_queue();

    ivmbb[id]->reset_node_counter();

    //make sure all are initialized
    int ret = pthread_barrier_wait(&barrier);
    if(ret==PTHREAD_BARRIER_SERIAL_THREAD)
    {
        updatedIntervals = 0;
        FILE_LOG(logDEBUG) << "=== start "<<get_num_threads()<<" exploration threads ===";
    }

    while (1) {
        //get global best UB
        pbb->best_found.getBest(bestCost);
        //set local UB
        ivmbb[id]->setBest(bestCost);

        if (allEnd.load()) {
            break;
        }else if (!ivmbb[id]->next()){
            request_work(id);
            thd_data[id]->has_work.store(ivmbb[id]->get_ivm()->beforeEnd());
        }else{
            try_answer_request(id);
        }

#ifdef WITH_MPI
        if(is_distributed())
        {
            if(pbb->workUpdateAvailable.load(std::memory_order_relaxed))
            {
                // FILE_LOG(logINFO) << "=== BREAK (update avail)";
                break;
            }
            if(atom_nb_steals>1)
            {
                // FILE_LOG(logINFO) << "=== BREAK (nb_steals "<<atom_nb_steals<<" )";
                break;
            }
            if(pbb->best_found.foundNewSolution){
                // FILE_LOG(logINFO) << "=== BREAK (new sol)";
                break;
            }

            bool passed=pbb->ttm->period_passed(WORKER_BALANCING);
            if(passed)
            {
                // FILE_LOG(logINFO) << "=== BREAK (time passed)";
                break;
            }
        }
#endif
    }

    allEnd.store(true);

    FILE_LOG(logDEBUG) << "=== Exit exploration loop";

    pbb->stats.totDecomposed += ivmbb[id]->get_decomposed_count();
    pbb->stats.leaves += ivmbb[id]->get_leaves_count();

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
        if( ivmbb[i]->get_ivm()->beforeEnd() ){
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

        if(ivmbb[i]->get_ivm()->beforeEnd())
        {
            memcpy(&ret[countTake*pbb->size],
                ivmbb[i]->get_ivm()->getNode().schedule.data(),
                pbb->size*sizeof(int)
            );
            countTake++;
        }
    }

    return nb;
}
