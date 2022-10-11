#ifndef POOLTHREAD_H_
#define POOLTHREAD_H_

#include "bbthread.h"
#include "poolbb.h"

class PoolThread : public bbthread
{
public:
    PoolThread(pbab* _pbb);
    // ~PoolThread();

    int
    shareWork(std::shared_ptr<PoolThread> thief_thread);

    void setLocalBest(const int best){};
    bool isEmpty(){
        return 0;
    };
    bool bbStep(){
        return 1;
    };
    void setRoot(const int *perm, int l1, int l2){} ;

};

#endif
