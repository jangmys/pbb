#ifndef BBTHREAD_H_
#define BBTHREAD_H_

#include <atomic>
#include <deque>

#include "log.h"


class pbab;

class bbthread
{
    friend class PoolController;
    friend class thread_controller;
    friend class matrix_controller;
    friend class worker_mc;
protected:
    pbab *pbb;

    virtual void setLocalBest(const int best) = 0;
    virtual bool isEmpty() = 0;
    virtual bool bbStep() = 0;
    virtual void setRoot(const int *perm, int l1, int l2) = 0;
public:
    bbthread(pbab * _pbb);
    ~bbthread();

    pthread_mutex_t mutex_ivm;
    // pthread_mutex_t mutex_workState;
    pthread_mutex_t mutex_workRequested;
    pthread_mutex_t mutex_shared;
    pthread_cond_t cond_shared;

    void set_work_state(const bool _b)
    {
        has_work.store(_b);
    }
    bool get_work_state()
    {
        return has_work.load();
    }
private:
    std::atomic<bool> has_work{false};
    bool received_work;

    std::deque<unsigned> request_queue;

    bool has_request();
    void reset_request_queue();

    void enqueue_request(unsigned id);
    size_t num_requests()
    {
        return request_queue.size();
    };
    unsigned get_oldest_request()
    {
        unsigned ret= request_queue.front();
        request_queue.pop_front();
        return ret;
    }

    void set_received_work(const bool _b)
    {
        received_work = _b;
    }
    bool has_received_work() const
    {
        return received_work;
    }
};

#endif // ifndef BBTHREAD_H_
