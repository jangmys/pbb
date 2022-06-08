#ifndef THREAD_CONTROLLER_H
#define THREAD_CONTROLLER_H

#include <atomic>
#include <memory>
#include <list>
#include <vector>
#include <iostream>

#include <victim_selector.h>
#include "bbthread.h"

class pbab;
class bbthread;

class thread_controller{
public:
    thread_controller(pbab * _pbb,int _nthreads);
    virtual ~thread_controller();

    void set_victim_select(std::unique_ptr<VictimSelector> _select)
    {
        victim_select = std::move(_select);
    }
protected:
    pbab* pbb;

    unsigned int get_num_threads();
    void interruptExploration();
    std::shared_ptr<bbthread>get_bbthread(int k);

    virtual std::shared_ptr<bbthread> make_bbexplorer() = 0;
    virtual int work_share(unsigned id, unsigned dest) = 0;

    std::vector<std::shared_ptr<bbthread>>bbb;

    std::atomic<unsigned int> atom_nb_explorers{0};
    std::atomic<unsigned int> atom_nb_steals{0};
    std::atomic<bool> allEnd{false};

    pthread_mutex_t mutex_end;
    pthread_barrier_t barrier;

    void counter_decrement();
    bool counter_increment(unsigned id);
    unsigned explorer_get_new_id();

    void push_request(unsigned victim, unsigned id);
    unsigned pull_request(unsigned id);
    void request_work(unsigned id);
    bool try_answer_request(unsigned id);
    void unlock_waiting_thread(unsigned id);

    void stop(unsigned id);
    void unlockWaiting(unsigned id);

    void resetExplorationState();

    std::unique_ptr<VictimSelector> victim_select;
private:
    unsigned M;

    std::atomic<unsigned int>end_counter{0};
};

#endif
