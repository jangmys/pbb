/*
request queue for multi-core BB

shared deque for handling inter-thread work requests.
using posix mutex to protect insert/retrieve operations

used in thread_controller.h
*/
#ifndef REQUEST_QUEUE_H_
#define REQUEST_QUEUE_H_

#include <pthread.h>
#include <deque>

#include "macros.h"


class RequestQueue
{
    friend class ThreadController;
public:
    RequestQueue(){
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);

        pthread_mutex_init(&mutex_queue, &attr);
        pthread_mutex_init(&mutex_shared, &attr);
        pthread_cond_init(&cond_shared, NULL);

        reset_request_queue();
    }

    ~RequestQueue(){
        pthread_mutex_destroy(&mutex_queue);
        pthread_mutex_destroy(&mutex_shared);
        pthread_cond_destroy(&cond_shared);
    };

    pthread_mutex_t mutex_shared;
    pthread_cond_t cond_shared;

    void reset_request_queue(){
        queue.clear();
    }
private:
    pthread_mutex_t mutex_queue;

    std::deque<unsigned> queue;

    bool has_request(){
        pthread_mutex_lock_check(&mutex_queue);
        bool ok = (queue.empty()) ? false : true;
        pthread_mutex_unlock(&mutex_queue);
        return ok;
    }

    void enqueue_request(unsigned id){
        pthread_mutex_lock_check(&mutex_queue);
        queue.push_back(id); // push ID into requestQueue
        pthread_mutex_unlock(&mutex_queue);
    }

    unsigned pop_front()
    {
        unsigned ret;

        pthread_mutex_lock_check(&mutex_queue);
        ret= queue.front();
        queue.pop_front();
        pthread_mutex_unlock(&mutex_queue);

        return ret;
    }
};

#endif // ifndef REQUEST_QUEUE_H_
