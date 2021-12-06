#include "macros.h"
#include "pbab.h"
#include "bbthread.h"

bbthread::bbthread(pbab * _pbb) : pbb(_pbb),received_work(false)
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

    reset_request_queue();

    FILE_LOG(logDEBUG) << " === bbthd constr1";
};

bbthread::~bbthread()
{
    pthread_mutex_destroy(&mutex_ivm);
    pthread_mutex_destroy(&mutex_workRequested);
    pthread_mutex_destroy(&mutex_workState);
    pthread_mutex_destroy(&mutex_shared);
}

void bbthread::reset_request_queue()
{
    request_queue.clear();
}

bool
bbthread::has_request()
{
    pthread_mutex_lock_check(&mutex_workRequested);
    bool ok = (request_queue.empty()) ? false : true;
    pthread_mutex_unlock(&mutex_workRequested);
    return ok;
}

void
bbthread::enqueue_request(unsigned id)
{
    request_queue.push_back(id); // push ID into requestQueue of victim thread
}
