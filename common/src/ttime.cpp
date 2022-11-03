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
    period_set(CHECKPOINT_TTIME, arguments::checkpointv);
    // std::cout<<"lasts 1 "<<arguments::checkpointv<<" "<<lasts[CHECKPOINT_TTIME]<<" "<<periods[CHECKPOINT_TTIME]<<std::endl;
    period_set(WORKER_BALANCING, arguments::balancingv);
    // std::cout<<"lasts 2 "<<lasts[WORKER_BALANCING]<<" "<<periods[WORKER_BALANCING]<<std::endl;

    lasts[TTIMEOUT]   = (time_t) time_get();// - (t * 1.0) * drand48());
    periods[TTIMEOUT] = arguments::timeout;

    processRequest = new mytimer();
    masterWalltime = new mytimer();

    wall   = new mytimer();
    update = new mytimer();
    split  = new mytimer();
    test   = new mytimer();

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

// void
// ttime::wait(int index)
// {
//     srand48(getpid());
//     double t = (unsigned int) (periods[index] * 1.0) * drand48();
//     std::cout << (unsigned int) t << std::endl << std::flush;
//
//     std::cout << "debut" << std::endl << std::flush;
//     sleep((unsigned int) t);
//     std::cout << "fin" << std::endl << std::flush;
// }

// ___________________________________________

void
ttime::period_set(int index, time_t t)
{
    srand48(getpid());
    lasts[index]   = (time_t) (time_get() - (t * 1.0) * drand48());
    periods[index] = t;
}

bool
ttime::period_passed(int index)
{
    time_t tmp = time_get();

    // std::cout<<"time? "<<tmp<<"\t"<<lasts[index]<<"\t"<<periods[index]<<std::endl<<std::flush;

    if ((tmp - lasts[index]) < periods[index]) return false;

    // multi-core worker threads execute this function...
    pthread_mutex_lock(&mutex_lasttime);
    lasts[index] = tmp;
    // std::cout<<"PASSED "<<lasts[index]<<"\n"<<std::flush;
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
        t2.tv_nsec += NSECS;
    }
    return t2;
}

timespec
ttime::addTime(struct timespec t1, struct timespec t2)
{
    t1.tv_sec  += t2.tv_sec;
    t1.tv_nsec += t2.tv_nsec;
    while (t1.tv_nsec > NSECS) {
        t1.tv_sec++;
        t1.tv_nsec -= NSECS;
    }
    return t1;
}
