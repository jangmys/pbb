#ifndef REQUEST_QUEUE_H_
#define REQUEST_QUEUE_H_

#include <atomic>
#include <deque>

#include "macros.h"

class pbab;

class RequestQueue
{
    friend class ThreadController;
public:
    RequestQueue(){
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);

        pthread_mutex_init(&mutex_workRequested, &attr);
        pthread_mutex_init(&mutex_shared, &attr);
        pthread_cond_init(&cond_shared, NULL);

        reset_request_queue();
    }

    ~RequestQueue(){
        pthread_mutex_destroy(&mutex_workRequested);
        pthread_mutex_destroy(&mutex_shared);
        pthread_cond_destroy(&cond_shared);
    };

    pthread_mutex_t mutex_workRequested;
    pthread_mutex_t mutex_shared;
    pthread_cond_t cond_shared;

    void reset_request_queue(){queue.clear();}
private:
    std::deque<unsigned> queue;

    bool has_request(){
        pthread_mutex_lock_check(&mutex_workRequested);
        bool ok = (queue.empty()) ? false : true;
        pthread_mutex_unlock(&mutex_workRequested);
        return ok;
    }

    void enqueue_request(unsigned id){
        queue.push_back(id); // push ID into requestQueue of victim thread
    }

    size_t num_requests()
    {
        return queue.size();
    };

    unsigned pop_front()
    {
        unsigned ret= queue.front();
        queue.pop_front();
        return ret;
    }
};

#endif // ifndef REQUEST_QUEUE_H_
