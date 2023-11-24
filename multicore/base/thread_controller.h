/*
base class for multi-core exploration, independent from data structure used for exploration (see ivm/matrix_controller, ll/pool_controller)
*/
#ifndef THREAD_CONTROLLER_H
#define THREAD_CONTROLLER_H

#include <atomic>
#include <memory>
#include <list>
#include <vector>
#include <iostream>

#include <victim_selector.h>
#include <thread_data.h>


class ThreadController{
public:
    ThreadController(pbab * _pbb,int _nthreads);
    virtual ~ThreadController();

    void set_victim_select(std::unique_ptr<VictimSelector> _select)
    {
        victim_select = std::move(_select);
    }
protected:
    pbab* pbb;
    unsigned M;
    std::vector<std::shared_ptr<RequestQueue>>thd_data;

    std::atomic<unsigned int> atom_nb_explorers{0};
    std::atomic<unsigned int> atom_nb_steals{0};
    std::atomic<bool> allEnd{false};
    std::atomic<unsigned int>end_counter{0};

    pthread_barrier_t barrier;

    virtual int work_share(unsigned id, unsigned dest) = 0;

    unsigned int get_num_threads();
    void interruptExploration();

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

};

#endif
