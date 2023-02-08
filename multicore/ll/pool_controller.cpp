
#include "poolbb.h"
#include "pool_controller.h"

#include "make_ll_algo.h"

PoolController::PoolController(pbab* pbb, int _nthreads) : ThreadController(pbb,_nthreads)
{
    llbb = std::vector<std::shared_ptr<Poolbb>>(get_num_threads(),nullptr);
}

void
PoolController::set_root(const int id, subproblem& node)
{
    // ibb[id]->llbb.set_root(node);
    llbb[id]->set_root(node);
}

int
PoolController::work_share(unsigned id, unsigned thief){
    assert(id != thief);
    assert(id < get_num_threads());
    assert(thief < get_num_threads());

    unsigned nb_share = llbb[id]->pool->size()/2;
    // unsigned nb_share = ibb[id]->llbb.pool->size()/2;

    if(nb_share > 1)
    {
        while(nb_share){
            llbb[thief]->pool->insert(
                llbb[id]->pool->back()
            );
            llbb[id]->pool->pop_back();
            nb_share--;
        }

        return 1;
    }
    return 0;
};



void
PoolController::explore_multicore()
{
    // get unique ID
    int id = explorer_get_new_id();
    // make explorer (if not already done)
    if(!llbb[id]){
        llbb[id] = make_poolbb(pbb);
        thd_data[id] = std::make_shared<RequestQueue>();
    }

    if( id == 0){
        subproblem tmp(pbb->size,pbb->best_found.initial_perm);

        set_root(0,tmp);
        thd_data[id]->has_work.store(true);
        // vec_has_work[id].store(true);
    }

    int ret = pthread_barrier_wait(&barrier);
    if(ret==PTHREAD_BARRIER_SERIAL_THREAD)
    {
        FILE_LOG(logDEBUG) << "=== start "<<get_num_threads()<<" exploration threads ===";
        std::cout<< "=== start "<<get_num_threads()<<" exploration threads ===\n"<<std::flush;
    }

    int local_best = INT_MAX;

    pbb->best_found.getBest(local_best);
    llbb[id]->set_local_best(local_best);

    while(1){
        if (allEnd.load()) {
            break;
        }
        else if (!llbb[id]->next()){
            request_work(id);
            thd_data[id]->has_work.store(!llbb[id]->isEmpty());
            // vec_has_work[id].store(!llbb[id]->isEmpty());
        }else{
            try_answer_request(id);
        }
    }

    allEnd.store(true);
    stop(id);

    pbb->stats.totDecomposed += llbb[id]->get_decomposed_count();
    pbb->stats.leaves += llbb[id]->get_leaves_count();
}

void*
poolbb_thread(void* _pc)
{
    PoolController* pc = (PoolController*)_pc;
    pc->explore_multicore();
    return NULL;
}


bool
PoolController::next()
{
    pthread_t threads[100];

    for(unsigned i=0; i<get_num_threads(); i++){
        pthread_create(&threads[i], NULL, poolbb_thread, (void*)this);
    }

    for(unsigned i=0; i<get_num_threads(); i++){
        int err = pthread_join(threads[i], NULL);
        if (err)
        {
            std::cout << "Failed to join Thread : " << strerror(err) << std::endl;
            return err;
        }
    }

    return true;
}
