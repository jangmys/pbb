// ======================================================
#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <pthread.h>
// ======================================================
#ifndef TTIME_H
#define TTIME_H

# define WORKER_BALANCING 0
# define CHECKPOINT_TTIME 1
# define TTIMEOUT       2
# define SIZE_TTIME       3

# define NSECS            1000000000

typedef struct mytimer{
    mytimer(){
        isOn            = false;
        start.tv_sec    = 0;
        start.tv_nsec   = 0;
        stop.tv_sec     = 0;
        stop.tv_nsec    = 0;
        elapsed.tv_sec  = 0;
        elapsed.tv_nsec = 0;
    };
    struct timespec start;
    struct timespec stop;
    struct timespec elapsed;
    bool            isOn;
}mytimer;

class ttime
{
public:
    time_t periods[SIZE_TTIME], lasts[SIZE_TTIME];

    ttime();
    ~ttime();

    void reset();

    static time_t
    time_get();
    void
    wait(int index);
    void
    period_set(int index, time_t t);
    bool
    period_passed(int index);

    timespec
    subtractTime(struct timespec t2, struct timespec t1);
    timespec
    addTime(struct timespec t1, struct timespec t2);

    void
    on(mytimer * t);
    void
    off(mytimer * t);
    void
    printElapsed(mytimer * t, const char * name);

    void
    logElapsed(mytimer * t, const char * name);

    float divide(mytimer *t1,mytimer *t2);
    float masterLoadPerc();
    // float getFloatTime(mytimer *t);

    mytimer * processRequest;
    mytimer * masterWalltime;
    mytimer * wall;
    mytimer * update;
    mytimer * split;
    mytimer * workerExploretime;
    mytimer * test;

    pthread_mutex_t mutex_lasttime;
};
#endif // ifndef TTIME_H
