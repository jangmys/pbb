#ifndef IVMTHREAD_H_
#define IVMTHREAD_H_

#include "pbab.h"
#include "bbthread.h"
#include "intervalbb.h"
#include "intervalbb_incr.h"
#include "intervalbb_easy.h"

static std::unique_ptr<Intervalbb<int>> make_interval_bb(pbab* pbb, unsigned bound_mode)
{
    if(arguments::boundMode == 0){
        return std::make_unique<Intervalbb<int>>(pbb);
    }else if(arguments::boundMode == 1){
        return std::make_unique<IntervalbbEasy<int>>(pbb);
    }else{
        return std::make_unique<IntervalbbIncr<int>>(pbb);
    }
}



class pbab;

class ivmthread : public bbthread
{
public:
    ivmthread(pbab* _pbb, std::unique_ptr<Intervalbb<int>> _ibb);
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
