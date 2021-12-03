#include <sys/sysinfo.h>
#include <assert.h>

#include "macros.h"
#include "arguments.h"
#include "pbab.h"
#include "thread_controller.h"

thread_controller::thread_controller(pbab * _pbb) : pbb(_pbb), size(pbb->size)
{
    // //set number of BB-explorers (threads)
    M = (arguments::nbivms_mc < 1) ? get_nprocs_conf() : arguments::nbivms_mc;

    if(arguments::singleNode){
        std::cout<<" === Single-node multi-core : Using "<<M<<" threads"<<std::endl;
        std::cout<<" === Problem size : "<<size<<std::endl;
    }

    for (int i = 0; i < (int) M; i++){
        bbb[i]=nullptr;
    }

    //barrier for syncing all explorer threads
    pthread_barrier_init(&barrier, NULL, M);

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);

    pthread_mutex_init(&mutex_end, &attr);

    //work stealing victim-list (only for "honest" strategy)
    pthread_mutex_init(&mutex_steal_list, &attr);
    for (unsigned i = 0; i < M; i++) {
        victim_list.push_back(i);
    }
    //for "random" WS strategy
    srand(time(NULL));

    atom_nb_explorers.store(0);// id_generator
    allEnd.store(false);
}

thread_controller::~thread_controller()
{
    pthread_mutex_destroy(&mutex_steal_list);
    pthread_mutex_destroy(&mutex_end);

}

void
thread_controller::counter_decrement()
{
    end_counter--;
}

bool
thread_controller::counter_increment(unsigned id)
{
    end_counter++;

    pthread_mutex_lock_check(&mutex_end);
    if (end_counter.load() == M) {
        allEnd.store(true);
        FILE_LOG(logDEBUG) << "++END COUNTER " << id << " " <<end_counter.load()<<" "<<M<<std::flush;
    }
    pthread_mutex_unlock(&mutex_end);

    return allEnd.load();
}

unsigned
thread_controller::explorer_get_new_id()
{
    assert(atom_nb_explorers.load() < MAX_EXPLORERS);
    return (atom_nb_explorers++);
}


unsigned
thread_controller::select_victim(unsigned id)
{
    // default: select left neighbor
    unsigned victim = (id == 0) ? (M - 1) : (id - 1);

    switch (arguments::mc_ws_select) {
        case 'r':
        {
            std::cout<<"ring\n";
            //ring selction is default
            break;
        }
        case 'a': {
            unsigned int attempts = 0;
            do {
                // randomly select active thread (at most nbIVM attempts...otherwise loop may be infinite)
                victim = rand() / (RAND_MAX /  M);
                if(++attempts > M)break;
            }while(victim == id || !bbb[victim]->getWorkState());
            // while (victim == id || !hasWork[victim]);

            return victim;

        }
        case 'o': {
            // select thread which has not made request for longest time
            pthread_mutex_lock(&mutex_steal_list);
            victim_list.remove(id);// remove id from list
            victim_list.push_back(id);// put at end
            victim = victim_list.front();// take first in list (oldest)

            // if(!hasWork[victim])
            if(!bbb[victim]->getWorkState())
                victim=(id == 0) ? (M - 1) : (id - 1);

            // FILE_LOG(logDEBUG4) << id << " list ...";
            // for(auto i:victim_list)
            // {
            //     FILE_LOG(logDEBUG4) << i;
            // }

            pthread_mutex_unlock(&mutex_steal_list);
            break;
        }
        default:
        {
            break;
        }
    }

    return victim;
}



void
thread_controller::push_request(unsigned victim, unsigned id)
{
    pthread_mutex_lock_check(&bbb[victim]->mutex_workRequested);
    (bbb[victim]->requestQueue).push_back(id); // push ID into requestQueue of victim thread
    pthread_mutex_unlock(&bbb[victim]->mutex_workRequested);
}


int
thread_controller::pull_request(unsigned id)
{
    int thief;

    pthread_mutex_lock_check(&bbb[id]->mutex_workRequested);
    //error check
    assert(bbb[id]->requestQueue.size()<M);

    // if(bbb[id]->requestQueue.size()>=M){
    //     FILE_LOG(logERROR) << "Received too many requests" << bbb[id]->requestQueue.size();
    //     for(auto i: bbb[id]->requestQueue) FILE_LOG(logERROR) << i;
    //     exit(-1);
    // }

    thief = (bbb[id]->requestQueue).front();
    (bbb[id]->requestQueue).pop_front();
    pthread_mutex_unlock(&bbb[id]->mutex_workRequested);
    return thief;
}

void
thread_controller::unlock_waiting_thread(unsigned id)
{
    //Note: For dependable use of condition variables, and to ensure that you do not lose wake-up operations on condition variables, your application should always use a Boolean predicate and a mutex with the condition variable.
    //https://www.ibm.com/support/knowledgecenter/en/ssw_ibm_i_74/apis/users_76.htm
    pthread_mutex_lock_check(&bbb[id]->mutex_shared);
    bbb[id]->setReceivedWork(true);
    pthread_cond_signal(&bbb[id]->cond_shared);
    pthread_mutex_unlock(&bbb[id]->mutex_shared);
}

void
thread_controller::request_work(unsigned id)
{
    bbb[id]->setWorkState(false);
    // hasWork[id]=false;
    if (counter_increment(id)) return;

    // if any pending requests, release waiting threads
    while (bbb[id]->has_request()) {
        unsigned thief = pull_request(id);

        if(thief==id){
            FILE_LOG(logERROR) << id << " try cancel myself " << thief;
            exit(-1);
        }

        counter_decrement();
        FILE_LOG(logDEBUG4) << id << " cancel " << thief << " count: "<<end_counter.load();

        unlock_waiting_thread(thief);
    }

    unsigned victim = select_victim(id); // select victim

    FILE_LOG(logDEBUG4) << id << " select " << victim << "\tcounter: "<<end_counter.load() << std::flush;

    if (victim != id) {
        pthread_mutex_lock_check(&bbb[id]->mutex_shared);
        bbb[id]->setReceivedWork(false);
        pthread_mutex_unlock(&bbb[id]->mutex_shared);

        push_request(victim, id);

        pthread_mutex_lock_check(&bbb[id]->mutex_shared);
        while (!bbb[id]->getReceivedWork() && !allEnd.load()) {
            pthread_cond_wait(&bbb[id]->cond_shared, &bbb[id]->mutex_shared);
        }
        pthread_mutex_unlock(&bbb[id]->mutex_shared);
    } else {
        FILE_LOG(logDEBUG4) << id << " selected myself " << victim;
    //     // cancel...
        counter_decrement();
    }

    bbb[id]->setWorkState(!bbb[id]->isEmpty());
    // hasWork[id]=!bbb[id]->isEmpty();

} // thread_controller::request_work

bool
thread_controller::try_answer_request(unsigned id)
{
    bool ret = false;

    //check if request queue empty ... called very often! some performance gain is possible by maintaining a separate boolean variable, but complicates code
    if(!bbb[id]->has_request())return false;

    unsigned thief = pull_request(id);

    if(bbb[thief]->getWorkState()){
    // if(hasWork[thief]){
        FILE_LOG(logERROR) << "id "<<id<<" FATAL error : thief "<<thief<<" got work";
        FILE_LOG(logERROR) << "and "<<bbb[id]->requestQueue.size()<<" pending requests";
        exit(-1);
    }

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
thread_controller::stop(unsigned id)
{
    unlockWaiting(id);

    int ret=pthread_barrier_wait(&barrier);
    if(ret==PTHREAD_BARRIER_SERIAL_THREAD){
        FILE_LOG(logDEBUG1) << "=== stop "<<M<<" ===";
    }
}
