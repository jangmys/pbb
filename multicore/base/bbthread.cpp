#include "macros.h"
#include "pbab.h"
#include "bbthread.h"

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
