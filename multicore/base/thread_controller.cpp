#include <sys/sysinfo.h>
#include <assert.h>

#include "macros.h"
#include "pbab.h"
#include "thread_controller.h"

ThreadController::ThreadController(pbab * _pbb, int _nthreads) : pbb(_pbb),M(_nthreads) //, size(pbb->size)
{
    requests = std::vector<RequestQueue>(M);

    vec_received_work = std::vector<std::atomic<bool>>(M);
    for (auto& b : vec_received_work) { std::atomic_init(&b, false); }

    vec_has_work = std::vector<std::atomic<bool>>(M);
    for (auto& b : vec_has_work) { std::atomic_init(&b, false); }

    threads = (pthread_t*)malloc(M*sizeof(threads));

    //barrier for syncing all explorer threads
    pthread_barrier_init(&barrier, NULL, M);

    atom_nb_explorers.store(0);// id_generator
    allEnd.store(false);
}

ThreadController::~ThreadController()
{
    free(threads);

    pthread_barrier_destroy(&barrier);
}

/* end_counter is atomic*/
void
ThreadController::counter_decrement()
{
    end_counter--;
    FILE_LOG(logDEBUG) << " DECREMENT COUNTER :" << end_counter<<std::flush;
}

/* end_counter is atomic*/
bool
ThreadController::counter_increment(unsigned id)
{
    end_counter++;

    FILE_LOG(logDEBUG) << "+++ "<<id<<" INCREMENT COUNTER :" << end_counter<<std::flush;

    if (end_counter.load() == M) {
        allEnd.store(true);
        FILE_LOG(logDEBUG) << "+++END COUNTER (" << id << ") VAL: " <<end_counter.load()<<"/"<<M<<std::flush;
        return true;
    }

    return false;
}

unsigned
ThreadController::explorer_get_new_id()
{
    // assert(atom_nb_explorers.load() < MAX_EXPLORERS);
    return (atom_nb_explorers++);
}


void
ThreadController::push_request(unsigned victim, unsigned id)
{
    requests[victim].enqueue_request(id);
}

unsigned
ThreadController::pull_request(unsigned id)
{
    unsigned thief = requests[id].pop_front();

    return thief;
}

void
ThreadController::unlock_waiting_thread(unsigned id)
{
    FILE_LOG(logDEBUG) << "=== Unlock ("<<id<<")";

    //Note: For dependable use of condition variables, and to ensure that you do not lose wake-up operations on condition variables, your application should always use a Boolean predicate and a mutex with the condition variable.
    //https://www.ibm.com/support/knowledgecenter/en/ssw_ibm_i_74/apis/users_76.htm
    pthread_mutex_lock_check(&requests[id].mutex_shared);
    vec_received_work[id].store(true);
    pthread_cond_signal(&requests[id].cond_shared);
    pthread_mutex_unlock(&requests[id].mutex_shared);

    FILE_LOG(logDEBUG) << "=== Unlocked ("<<id<<")";
}

void
ThreadController::request_work(unsigned id)
{
    vec_has_work[id].store(false);
    if (counter_increment(id)) return;

    // if any pending requests, release waiting threads
    while (requests[id].has_request()) {
        unsigned thief = pull_request(id);
        assert(thief != id);

        counter_decrement();
        FILE_LOG(logDEBUG) << id << " cancel " << thief << " count: "<<end_counter.load();

        unlock_waiting_thread(thief);
    }

    unsigned victim = 0;

    while(!vec_has_work[victim].load() && !allEnd.load())
        victim = (*victim_select)(id);

    FILE_LOG(logDEBUG4) << id << " select " << victim << "\tcounter: "<<end_counter.load() << std::flush;

    if (victim != id && vec_has_work[victim].load()) {
        pthread_mutex_lock_check(&requests[id].mutex_shared);
        vec_received_work[id].store(false);
        pthread_mutex_unlock(&requests[id].mutex_shared);

        push_request(victim,id);

        pthread_mutex_lock_check(&requests[id].mutex_shared);
        while (!vec_received_work[id].load() && !allEnd.load()) {
            pthread_cond_wait(&requests[id].cond_shared, &requests[id].mutex_shared);
        }
        vec_received_work[id].store(false);
        pthread_mutex_unlock(&requests[id].mutex_shared);
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
    // if(!requests[id].has_request())return false;
    if(!requests[id].has_request())return false;

    unsigned thief = pull_request(id);

    assert(!vec_has_work[thief].load());

    atom_nb_steals += work_share(id, thief);

    counter_decrement();
    FILE_LOG(logDEBUG) << id << " answer " << thief << " counter: " << end_counter.load() << std::flush;

    unlock_waiting_thread(thief);

    return ret;
} // ThreadController::try_answer_request


void
ThreadController::unlockWaiting(unsigned id)
{
    FILE_LOG(logDEBUG) << "=== Unlock all waiting ("<<id<<")";

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
