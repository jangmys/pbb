#ifndef IVMTHREAD_H_
#define IVMTHREAD_H_

#include "pbab.h"
#include "bbthread.h"
#include "intervalbb.h"

class pbab;

class ivmthread : public bbthread
{
public:
    ivmthread(pbab* _pbb, std::shared_ptr<Intervalbb<int>> _ibb);
    ~ivmthread();

    std::shared_ptr<Intervalbb<int>> ivmbb;

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
    void setRoot(const int* perm, int l1, int l2);

    bool isEmpty(){
        return !ivmbb->get_ivm()->beforeEnd();
    }
};

#endif
