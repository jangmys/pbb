#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>


#include "../include/ttime.h"
#include "../include/arguments.h"
#include "../include/log.h"



// ___________________________________________
ttime::ttime()
{
    period_set(T_CHECKPOINT, arguments::checkpointv);
    period_set(T_WORKER_BALANCING, arguments::balancingv);

    FILE_LOG(logINFO) << "Checkpointing interval: "<<periods[T_CHECKPOINT]<<" sec, starting from "<<lasts[T_CHECKPOINT];
    FILE_LOG(logINFO) << "Balancing interval: "<<periods[T_WORKER_BALANCING]<<" sec, starting from "<<lasts[T_WORKER_BALANCING];

    lasts[T_TIMEOUT]   = (time_t) time_get();// - (t * 1.0) * drand48());
    periods[T_TIMEOUT] = arguments::timeout;

    processRequest = new mytimer();
    masterWalltime = new mytimer();

    wall   = new mytimer();
    update = new mytimer();
    split  = new mytimer();

    workerExploretime = new mytimer();

    pthread_mutex_init(&mutex_lasttime, NULL);
}

ttime::~ttime()
{
    delete processRequest;
    delete masterWalltime;
    delete wall;
    delete update;
    delete split;

    pthread_mutex_destroy(&mutex_lasttime);
}

void ttime::reset()
{
    wall->isOn=false;
    wall->elapsed.tv_sec=0;
    wall->elapsed.tv_nsec=0;

    processRequest->isOn=false;
    processRequest->elapsed.tv_sec=0;
    processRequest->elapsed.tv_nsec=0;
    //     wall.tv_nsec=0;
    masterWalltime->isOn=false;
    masterWalltime->elapsed.tv_sec=0;
    masterWalltime->elapsed.tv_nsec=0;
}

time_t
ttime::time_get()
{
    time_t tmp;

    time(&tmp);
    return tmp;
}


void
ttime::period_set(TTimeIndex index, time_t t)
{
    //sets period[index] to t and
    //initializes lasts[index] to NOW (randomly delayed in the past)
    srand48(getpid());
    lasts[index]   = (time_t) (time_get() - (t * 1.0) * drand48());
    periods[index] = t;
}

bool
ttime::period_passed(TTimeIndex index)
{
    time_t tmp = time_get();

    // std::cout<<"time? "<<tmp<<"\t"<<lasts[index]<<"\t"<<periods[index]<<std::endl<<std::flush;

    if ((tmp - lasts[index]) < periods[index]) return false;

    // multi-core worker threads execute this function...
    pthread_mutex_lock(&mutex_lasttime);
    lasts[index] = tmp;
    pthread_mutex_unlock(&mutex_lasttime);

    return true;
}

// ======================================
void
ttime::on(mytimer * t)
{
    t->isOn = true;
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

void
ttime::off(mytimer * t)
{
    clock_gettime(CLOCK_MONOTONIC, &t->stop);

    struct timespec diff = subtractTime(t->stop, t->start);
    if (t->isOn) t->elapsed = addTime(t->elapsed, diff);
    t->isOn = false;
}

void
ttime::printElapsed(mytimer * t, const char * name)
{
    printf("%s\t:\t %lld.%.9ld\n", name, (long long) t->elapsed.tv_sec, t->elapsed.tv_nsec);
}

void
ttime::logElapsed(mytimer * t, const char * name)
{
    FILE_LOG(logINFO) << name << "\t:" << (long long) t->elapsed.tv_sec << "." << t->elapsed.tv_nsec;
}

float
ttime::divide(mytimer * t1, mytimer * t2)
{
    return (t1->elapsed.tv_sec + t1->elapsed.tv_nsec / 1e9) / (t2->elapsed.tv_sec + t2->elapsed.tv_nsec / 1e9);
}

float
ttime::masterLoadPerc()
{
    return 100.0 * divide(masterWalltime, wall);
}

timespec
ttime::subtractTime(struct timespec t2, struct timespec t1)
{
    t2.tv_sec  -= t1.tv_sec;
    t2.tv_nsec -= t1.tv_nsec;
    while (t2.tv_nsec < 0) {
        t2.tv_sec--;
        t2.tv_nsec += nsecs_per_sec;
    }
    return t2;
}

timespec
ttime::addTime(struct timespec t1, struct timespec t2)
{
    t1.tv_sec  += t2.tv_sec;
    t1.tv_nsec += t2.tv_nsec;
    while (t1.tv_nsec > nsecs_per_sec) {
        t1.tv_sec++;
        t1.tv_nsec -= nsecs_per_sec;
    }
    return t1;
}
