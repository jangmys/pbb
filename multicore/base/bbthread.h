#ifndef BBTHREAD_H_
#define BBTHREAD_H_

#include <deque>

#include "log.h"
#include "../ivm/sequentialbb.h"

class pbab;

class bbthread
{
public:
    bbthread(pbab * _pbb) : pbb(_pbb),receivedWork(false),got_work(false)
    {
        // FILE_LOG(logDEBUG) << " === bbthd constr0";
        // pbb=_pbb;

        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);

        pthread_mutex_init(&mutex_ivm, &attr);
        pthread_mutex_init(&mutex_workRequested, &attr);
        pthread_mutex_init(&mutex_workState, &attr);

        requestQueue.clear();

        pthread_mutex_init(&mutex_shared, &attr);
        pthread_cond_init(&cond_shared, NULL);

        FILE_LOG(logDEBUG) << " === bbthd constr1";
    };

    ~bbthread()
    {
        pthread_mutex_destroy(&mutex_ivm);
        pthread_mutex_destroy(&mutex_workRequested);
        pthread_mutex_destroy(&mutex_workState);
    }

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
        // receivedWork = _b;
    }
    bool getWorkState()
    {
        return has_work.load();
        // return receivedWork;
    }


    std::atomic<bool> has_work;
protected:
    pbab *pbb;

    bool receivedWork;
    bool got_work;
};

#endif // ifndef BBTHREAD_H_
