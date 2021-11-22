#ifndef IVMTHREAD_H_
#define IVMTHREAD_H_

#include "pbab.h"
#include "bbthread.h"
#include "sequentialbb.h"

class pbab;

class ivmthread : public bbthread
{
public:
    ivmthread(pbab* _pbb);
    ~ivmthread();

    sequentialbb<int>* ivmbb;

    void
    getSchedule(int *sch)
    {
        ivmbb->IVM->getSchedule(sch);
    }

    int
    shareWork(int numerator, int denominator, sequentialbb<int> *thief_thread);

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
