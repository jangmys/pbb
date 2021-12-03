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

    void
    getSchedule(int *sch)
    {
        ivmbb->IVM->getSchedule(sch);
    }

    int
    shareWork(std::shared_ptr<sequentialbb<int>> thief_thread);

    bool bbStep();
    void setRoot(const int* perm);

    bool isEmpty(){
        return !ivmbb->IVM->beforeEnd();
    }
};

#endif
