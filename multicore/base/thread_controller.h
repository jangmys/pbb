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

class thread_controller{
public:
    thread_controller(pbab * _pbb);
    virtual ~thread_controller();

protected:
    pbab* pbb;

    unsigned int get_num_threads();
    void interruptExploration();
    std::shared_ptr<bbthread>get_bbthread(int k);

    virtual std::shared_ptr<bbthread> make_bbexplorer() = 0;
    virtual int work_share(unsigned id, unsigned dest) = 0;

    std::shared_ptr<bbthread>bbb[MAX_EXPLORERS];

    std::atomic<unsigned int> atom_nb_explorers{0};
    std::atomic<unsigned int> atom_nb_steals{0};
    std::atomic<bool> allEnd{false};

    pthread_mutex_t mutex_end;
    pthread_barrier_t barrier;
    pthread_mutex_t mutex_steal_list;
    std::list<int> victim_list;

    void counter_decrement();
    bool counter_increment(unsigned id);
    unsigned explorer_get_new_id();

    unsigned select_victim(unsigned id);

    void push_request(unsigned victim, unsigned id);
    unsigned pull_request(unsigned id);
    void request_work(unsigned id);
    bool try_answer_request(unsigned id);
    void unlock_waiting_thread(unsigned id);

    void stop(unsigned id);
    void unlockWaiting(unsigned id);

    void resetExplorationState();
private:
    unsigned M;

    std::atomic<unsigned int>end_counter{0};
};

#endif
