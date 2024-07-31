#include <sys/sysinfo.h>
#include <assert.h>

#include "macros.h"
#include "pbab.h"
#include "thread_controller.h"

ThreadController::ThreadController(pbab * _pbb, int _nthreads,int _worker_rank/*=0*/) : 
	pbb(_pbb),
	M(_nthreads),
	local_mpi_rank(_worker_rank),
	thd_data(std::vector< std::shared_ptr<RequestQueue> >(_nthreads,nullptr)),
    	victim_select(std::make_shared<RandomVictimSelector>(_nthreads))
{
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    std::cout<<"LRank : "<<local_mpi_rank<<" Got "<<num_cores<<" cores for "<<M<<" workers\n";

    //barrier for syncing all explorer threads
    pthread_barrier_init(&barrier, NULL, M);
}

ThreadController::~ThreadController()
{
    pthread_barrier_destroy(&barrier);
}

/* end_counter is atomic*/
void
ThreadController::counter_decrement()
{
    end_counter--;
}

/* end_counter is atomic*/
bool
ThreadController::counter_increment(unsigned id)
{
    end_counter++;

    if (end_counter.load() == M) {
        allEnd.store(true);
        return true;
    }

    return false;
}

unsigned
ThreadController::explorer_get_new_id()
{
    return (atom_nb_explorers++);
}


void
ThreadController::push_request(unsigned victim, unsigned id)
{
    thd_data[victim]->enqueue_request(id);
}

unsigned
ThreadController::pull_request(unsigned id)
{
    unsigned thief = thd_data[id]->pop_front();

    return thief;
}

void
ThreadController::unlock_waiting_thread(unsigned id)
{
    //Note: For dependable use of condition variables, and to ensure that you do not lose wake-up operations on condition variables, your application should always use a Boolean predicate and a mutex with the condition variable.
    //https://www.ibm.com/support/knowledgecenter/en/ssw_ibm_i_74/apis/users_76.htm
    pthread_mutex_lock_check(&thd_data[id]->mutex_shared);
    thd_data[id]->received_work.store(true);
    pthread_cond_signal(&thd_data[id]->cond_shared);
    pthread_mutex_unlock(&thd_data[id]->mutex_shared);
}

void
ThreadController::request_work(unsigned id)
{
    thd_data[id]->has_work.store(false);
    if (counter_increment(id)) return;

    // if any pending requests, release waiting threads
    while (thd_data[id]->has_request()) {
        unsigned thief = pull_request(id);
        assert(thief != id);

        counter_decrement();
        unlock_waiting_thread(thief);
    }

    unsigned victim = 0;
    do{
        victim = (*victim_select)(id);

        if((victim < 0)||(victim >=M)){
            std::cout<<"fatal error : victim id "<<victim<<" out of bounds (0-"<<M<<")\n";
            exit(-1);
        }
        if(!thd_data[victim]){
            std::cout<<"fatal error : victim thread data uninitialized\n";
            exit(-1);
        }
    }while(!thd_data[victim]->has_work.load() && !allEnd.load());

    FILE_LOG(logDEBUG4) << id << " select " << victim << "\tcounter: "<<end_counter.load() << std::flush;

    if (victim != id && thd_data[victim]->has_work.load()) {
        pthread_mutex_lock_check(&thd_data[id]->mutex_shared);
        thd_data[id]->received_work.store(false);
        pthread_mutex_unlock(&thd_data[id]->mutex_shared);

        push_request(victim,id);

        pthread_mutex_lock_check(&thd_data[id]->mutex_shared);
        while (!thd_data[id]->received_work.load() && !allEnd.load()) {
            pthread_cond_wait(&thd_data[id]->cond_shared, &thd_data[id]->mutex_shared);
        }
        thd_data[id]->received_work.store(false);
        pthread_mutex_unlock(&thd_data[id]->mutex_shared);
    } else {
        // cancel...
        counter_decrement();
    }
} // ThreadController::request_work

bool
ThreadController::try_answer_request(unsigned id)
{
    bool ret = false;
    //check if request queue empty ... called very often! some performance gain is possible by maintaining a separate boolean variable, but complicates code
    if(!thd_data[id]->has_request())return false;

    unsigned thief = pull_request(id);

    assert(!thd_data[thief]->has_work.load());

    atom_nb_steals += work_share(id, thief);

    counter_decrement();
    unlock_waiting_thread(thief);

    return ret;
} // ThreadController::try_answer_request

void
ThreadController::unlockWaiting(unsigned id)
{
    // unlock threads waiting for work
    for (unsigned i = 0; i < M; i++) {
        if (i == id) continue;
        unlock_waiting_thread(i);
    }
}

void
ThreadController::resetExplorationState()
{
    //reset global variables
    allEnd.store(false);
    end_counter.store(0);// termination counter
    atom_nb_explorers.store(0);// id_generator
    atom_nb_steals.store(0);//count work thefts
}

void
ThreadController::interruptExploration()
{
    allEnd.store(true);
}

unsigned int
ThreadController::get_num_threads()
{
    return M;
}

void
ThreadController::stop(unsigned id)
{
    unlockWaiting(id);

    (void)pthread_barrier_wait(&barrier);
}
