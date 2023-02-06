#ifndef THREAD_CONTROLLER_H
#define THREAD_CONTROLLER_H

#include <atomic>
#include <memory>
#include <list>
#include <vector>
#include <iostream>

#include <victim_selector.h>
#include <request_queue.h>


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
    pthread_t *threads;

    unsigned int get_num_threads();
    void interruptExploration();

    virtual int work_share(unsigned id, unsigned dest) = 0;

    std::vector<RequestQueue> requests;

    std::atomic<unsigned int> atom_nb_explorers{0};
    std::atomic<unsigned int> atom_nb_steals{0};
    std::atomic<bool> allEnd{false};

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
    std::vector<std::atomic<bool>> vec_received_work;
    std::vector<std::atomic<bool>> vec_has_work;
private:
    unsigned M;

    std::atomic<unsigned int>end_counter{0};
};

#endif
