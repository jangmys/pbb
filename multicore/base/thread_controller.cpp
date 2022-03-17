#include <sys/sysinfo.h>
#include <assert.h>

#include "macros.h"
#include "arguments.h"
#include "pbab.h"
#include "thread_controller.h"

thread_controller::thread_controller(pbab * _pbb, int _nthreads) : pbb(_pbb),M(_nthreads) //, size(pbb->size)
{
    //set number of BB-explorers (threads)
    bbb = std::vector<std::shared_ptr<bbthread>>(M,nullptr);

    //barrier for syncing all explorer threads
    pthread_barrier_init(&barrier, NULL, M);

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);

    pthread_mutex_init(&mutex_end, &attr);

    //work stealing victim-list (only for "honest" strategy)
    // pthread_mutex_init(&mutex_steal_list, &attr);
    // for (unsigned i = 0; i < M; i++) {
    //     victim_list.push_back(i);
    // }
    //for "random" WS strategy
    srand(time(NULL));

    atom_nb_explorers.store(0);// id_generator
    allEnd.store(false);

    victim_select = std::make_unique<RingVictimSelector>(M);

    if(arguments::singleNode){
        std::cout<<" === Single-node multi-core : Using "<<M<<" threads"<<std::endl;
        std::cout<<" === Problem size : "<<pbb->size<<std::endl;
    }
}

thread_controller::~thread_controller()
{
    // pthread_mutex_destroy(&mutex_steal_list);
    pthread_mutex_destroy(&mutex_end);
    pthread_barrier_destroy(&barrier);
}

std::shared_ptr<bbthread>
thread_controller::get_bbthread(int k)
{
    return bbb[k];
};

void
thread_controller::counter_decrement()
{
    end_counter--;
}

bool
thread_controller::counter_increment(unsigned id)
{
    end_counter++;

    // pthread_mutex_lock_check(&mutex_end);
    if (end_counter.load() == M) {
        allEnd.store(true);
        FILE_LOG(logDEBUG) << "++END COUNTER " << id << " " <<end_counter.load()<<" "<<M<<std::flush;
    }
    // pthread_mutex_unlock(&mutex_end);

    return allEnd.load();
}

unsigned
thread_controller::explorer_get_new_id()
{
    assert(atom_nb_explorers.load() < MAX_EXPLORERS);
    return (atom_nb_explorers++);
}


// unsigned
// thread_controller::select_victim(unsigned id)
// {
    // default: select left neighbor
    // unsigned victim = (id == 0) ? (M - 1) : (id - 1);

    // return (*victim_select)(id,bbb);

    // switch (arguments::mc_ws_select) {
    //     case 'r':
    //     {
    //         // std::cout<<"ring\n";
    //         //ring selction is default
    //         break;
    //     }
    //     case 'a': {
    //         unsigned int attempts = 0;
    //         do {
    //             // randomly select active thread (at most nbIVM attempts...otherwise loop may be infinite)
    //             victim = rand() / (RAND_MAX /  M);
    //             if(++attempts > M)break;
    //         }while(victim == id || !bbb[victim]->get_work_state());
    //
    //         return victim;
    //     }
    //     case 'o': {
    //         // select thread which has not made request for longest time
    //         pthread_mutex_lock(&mutex_steal_list);
    //         victim_list.remove(id);// remove id from list
    //         victim_list.push_back(id);// put at end
    //         victim = victim_list.front();// take first in list (oldest)
    //
    //         if(!bbb[victim]->get_work_state())
    //             victim=(id == 0) ? (M - 1) : (id - 1);
    //
    //         pthread_mutex_unlock(&mutex_steal_list);
    //         break;
    //     }
    //     default:
    //     {
    //         break;
    //     }
    // }

    // return victim;
// }



void
thread_controller::push_request(unsigned victim, unsigned id)
{
    pthread_mutex_lock_check(&bbb[victim]->mutex_workRequested);
    bbb[victim]->enqueue_request(id);
    pthread_mutex_unlock(&bbb[victim]->mutex_workRequested);
}

unsigned
thread_controller::pull_request(unsigned id)
{
    unsigned thief;

    pthread_mutex_lock_check(&bbb[id]->mutex_workRequested);
    assert(bbb[id]->num_requests()<M);
    thief = bbb[id]->get_oldest_request();
    pthread_mutex_unlock(&bbb[id]->mutex_workRequested);
    return thief;
}

void
thread_controller::unlock_waiting_thread(unsigned id)
{
    //Note: For dependable use of condition variables, and to ensure that you do not lose wake-up operations on condition variables, your application should always use a Boolean predicate and a mutex with the condition variable.
    //https://www.ibm.com/support/knowledgecenter/en/ssw_ibm_i_74/apis/users_76.htm
    pthread_mutex_lock_check(&bbb[id]->mutex_shared);
    bbb[id]->set_received_work(true);
    pthread_cond_signal(&bbb[id]->cond_shared);
    pthread_mutex_unlock(&bbb[id]->mutex_shared);
}

void
thread_controller::request_work(unsigned id)
{
    bbb[id]->set_work_state(false);
    // hasWork[id]=false;
    if (counter_increment(id)) return;

    // if any pending requests, release waiting threads
    while (bbb[id]->has_request()) {
        unsigned thief = pull_request(id);
        assert(thief != id);

        counter_decrement();
        FILE_LOG(logDEBUG4) << id << " cancel " << thief << " count: "<<end_counter.load();

        unlock_waiting_thread(thief);
    }

    unsigned victim = (*victim_select)(id,bbb);
    // unsigned victim = select_victim(id); // select victim

    FILE_LOG(logDEBUG4) << id << " select " << victim << "\tcounter: "<<end_counter.load() << std::flush;

    if (victim != id) {
        pthread_mutex_lock_check(&bbb[id]->mutex_shared);
        bbb[id]->set_received_work(false);
        pthread_mutex_unlock(&bbb[id]->mutex_shared);

        push_request(victim, id);

        pthread_mutex_lock_check(&bbb[id]->mutex_shared);
        while (!bbb[id]->has_received_work() && !allEnd.load()) {
            pthread_cond_wait(&bbb[id]->cond_shared, &bbb[id]->mutex_shared);
        }
        pthread_mutex_unlock(&bbb[id]->mutex_shared);
    } else {
        FILE_LOG(logDEBUG4) << id << " selected myself " << victim;
    //     // cancel...
        counter_decrement();
    }

    bbb[id]->set_work_state(!bbb[id]->isEmpty());
    // hasWork[id]=!bbb[id]->isEmpty();

} // thread_controller::request_work

bool
thread_controller::try_answer_request(unsigned id)
{
    bool ret = false;

    //check if request queue empty ... called very often! some performance gain is possible by maintaining a separate boolean variable, but complicates code
    if(!bbb[id]->has_request())return false;

    unsigned thief = pull_request(id);

    assert(!bbb[thief]->get_work_state());

    pthread_mutex_lock_check(&bbb[thief]->mutex_ivm);
    atom_nb_steals += work_share(id, thief);
    pthread_mutex_unlock(&bbb[thief]->mutex_ivm);

    counter_decrement();
    FILE_LOG(logDEBUG4) << id << " answer " << thief << " counter: " << end_counter.load() << std::flush;

    unlock_waiting_thread(thief);

    return ret;
} // thread_controller::try_answer_request

void
thread_controller::unlockWaiting(unsigned id)
{
    FILE_LOG(logDEBUG1) << "Unlock all waiting";

    // unlock threads waiting for work
    for (unsigned i = 0; i < M; i++) {
        if (i == id) continue;
        unlock_waiting_thread(i);
    }
}

void
thread_controller::resetExplorationState()
{
    //reset global variables
    end_counter.store(0);// termination counter
    allEnd.store(false);
    atom_nb_explorers.store(0);// id_generator
    atom_nb_steals.store(0);//count work thefts
}

void
thread_controller::interruptExploration()
{
    allEnd.store(true);
}

unsigned int
thread_controller::get_num_threads()
{
    return M;
}

void
thread_controller::stop(unsigned id)
{
    unlockWaiting(id);

    int ret=pthread_barrier_wait(&barrier);
    if(ret==PTHREAD_BARRIER_SERIAL_THREAD){
        FILE_LOG(logDEBUG1) << "=== stop "<<M<<" ===";
    }
}
