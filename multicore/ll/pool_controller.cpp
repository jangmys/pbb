
#include "poolbb.h"
#include "pool_controller.h"

PoolController::PoolController(pbab* pbb, int _nthreads) : thread_controller(pbb,_nthreads)
{

}

std::shared_ptr<bbthread>
PoolController::make_bbexplorer(){
    return std::make_shared<PoolThread>(
        pbb
    );
};

void
PoolController::explore_multicore()
{
    // get unique ID
    int id = explorer_get_new_id();
    // make explorer (if not already done)
    if(!bbb[id]){
        std::cout<<"hello\t"<<id<<"\n";
        bbb[id] = make_bbexplorer();
    }

    int ret = pthread_barrier_wait(&barrier);
    if(ret==PTHREAD_BARRIER_SERIAL_THREAD)
    {
        FILE_LOG(logDEBUG) << "=== start "<<get_num_threads()<<" exploration threads ===";

        std::cout<< "=== start "<<get_num_threads()<<" exploration threads ===\n";
    }

    // while()
    int local_best = INT_MAX;

    pbb->sltn->getBest(local_best);
    bbb[id]->setLocalBest(local_best);

    bbb[id]->bbStep();




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
}
