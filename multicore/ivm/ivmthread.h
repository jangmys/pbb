#ifndef IVMTHREAD_H_
#define IVMTHREAD_H_

#include "pbab.h"
#include "bbthread.h"

class pbab;

class ivmthread : public bbthread
{
public:
    ivmthread(pbab* _pbb) :
        bbthread(_pbb),
        ivmbb(new sequentialbb(_pbb,_pbb->size))
    {
    };
    ~ivmthread(){   };

    sequentialbb* ivmbb;

    void
    getSchedule(int *sch)
    {
        ivmbb->bd->getSchedule(sch);
    }

    int
    shareWork(int numerator, int denominator, sequentialbb *thief_thread);

    bool bbStep(){
        return ivmbb->next();
    }

    void setRoot(const int* perm){
        ivmbb->setRoot(perm);
    }

    bool isEmpty(){
        return !ivmbb->IVM->beforeEnd();
    }
};

#endif
