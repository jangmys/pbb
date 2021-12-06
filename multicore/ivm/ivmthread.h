#ifndef IVMTHREAD_H_
#define IVMTHREAD_H_

#include "pbab.h"
#include "bbthread.h"
#include "sequentialbb.h"

class pbab;

class ivmthread : public bbthread
{
public:
    explicit ivmthread(pbab* _pbb);
    ~ivmthread();

    std::shared_ptr<sequentialbb<int>> ivmbb;

    subproblem&
    getNode()
    {
        return ivmbb->get_ivm()->getNode();
    }

    void
    getInterval(int *pos,int *end){
        ivmbb->get_ivm()->getInterval(pos, end);
    }

    int
    shareWork(std::shared_ptr<ivmthread> thief_thread);

    //pure virtuals
    void setLocalBest(const int best){
        ivmbb->setBest(best);
    }

    bool bbStep();
    void setRoot(const int* perm);

    bool isEmpty(){
        return !ivmbb->get_ivm()->beforeEnd();
    }
};

#endif
