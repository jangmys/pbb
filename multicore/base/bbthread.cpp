#include "macros.h"
#include "pbab.h"
#include "bbthread.h"

bbthread::bbthread(pbab * _pbb) : pbb(_pbb),receivedWork(false),got_work(false)
{
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);

    //
    pthread_mutex_init(&mutex_ivm, &attr);
    pthread_mutex_init(&mutex_workRequested, &attr);
    pthread_mutex_init(&mutex_workState, &attr);

    pthread_mutex_init(&mutex_shared, &attr);
    pthread_cond_init(&cond_shared, NULL);

    requestQueue.clear();

    FILE_LOG(logDEBUG) << " === bbthd constr1";
};

bbthread::~bbthread()
{
    pthread_mutex_destroy(&mutex_ivm);
    pthread_mutex_destroy(&mutex_workRequested);
    pthread_mutex_destroy(&mutex_workState);
    pthread_mutex_destroy(&mutex_shared);
}

void bbthread::reset_requestQueue()
{
    requestQueue.clear();
}

bool
bbthread::has_request()
{
    pthread_mutex_lock_check(&mutex_workRequested);
    bool ok = (requestQueue.empty()) ? false : true;
    pthread_mutex_unlock(&mutex_workRequested);
    return ok;
}
