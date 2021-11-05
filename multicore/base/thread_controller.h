#ifndef THREAD_CONTROLLER_H
#define THREAD_CONTROLLER_H

#include <atomic>
#include <memory>
#include <list>
#include <vector>

#include "bbthread.h"

#define MAX_EXPLORERS 100

class pbab;
class bbthread;

static pthread_mutex_t instance_mutex = PTHREAD_MUTEX_INITIALIZER;

class thread_controller{
public:
    thread_controller(pbab * _pbb);
    virtual ~thread_controller();

protected:
    pbab* pbb;
    int size;
    unsigned M;

    virtual bbthread* make_bbexplorer(unsigned _id) = 0;
    virtual int work_share(unsigned id, unsigned dest) = 0;
    // bbthread *sbb[MAX_EXPLORERS];
    bbthread *bbb[MAX_EXPLORERS];

    std::atomic<unsigned int> atom_nb_explorers{0};
    std::atomic<unsigned int> atom_nb_steals{0};
    std::atomic<unsigned int>end_counter{0};
    std::atomic<bool> allEnd;

    // std::atomic_flag stop_exploring = ATOMIC_FLAG_INIT;

    // std::vector<bool>hasWork;

    pthread_mutex_t mutex_end;

    pthread_barrier_t barrier;
    pthread_mutex_t mutex_steal_list;
    std::list<int> victim_list;

    void counter_decrement();
    bool counter_increment(unsigned id);
    unsigned explorer_get_new_id();

    unsigned select_victim(unsigned id);

    void push_request(unsigned victim, unsigned id);
    int pull_request(unsigned id);
    void request_work(unsigned id);
    bool try_answer_request(unsigned id);
    void unlock_waiting_thread(unsigned id);

    void stop(unsigned id);
    void unlockWaiting(unsigned id);
};

#endif
