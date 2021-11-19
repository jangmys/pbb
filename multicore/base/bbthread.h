#ifndef BBTHREAD_H_
#define BBTHREAD_H_

#include <deque>

#include "log.h"
#include "../ivm/sequentialbb.h"

class pbab;

class bbthread
{
protected:
    pbab *pbb;

    bool receivedWork;
    bool got_work;
public:
    bbthread(pbab * _pbb);
    ~bbthread();

    virtual bool isEmpty() = 0;
    virtual bool bbStep() = 0;
    virtual void setRoot(const int *perm) = 0;

    std::deque<int> requestQueue;

    pthread_mutex_t mutex_ivm;
    pthread_mutex_t mutex_workState;
    pthread_mutex_t mutex_workRequested;
    pthread_mutex_t mutex_shared;

    pthread_cond_t cond_shared;

    bool has_request();
    void reset_requestQueue();

    void setReceivedWork(const bool _b)
    {
        receivedWork = _b;
    }
    bool getReceivedWork()
    {
        return receivedWork;
    }

    void setWorkState(const bool _b)
    {
        has_work.store(_b);
    }
    bool getWorkState()
    {
        return has_work.load();
    }

    std::atomic<bool> has_work;
};

#endif // ifndef BBTHREAD_H_
