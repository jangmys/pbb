#include "ivmthread.h"

ivmthread::ivmthread(pbab* _pbb) :
    bbthread(_pbb),
    ivmbb(std::make_shared<sequentialbb<int>>(_pbb,_pbb->size))
{
};


ivmthread::~ivmthread()
{   };


bool ivmthread::bbStep(){
    return ivmbb->next();
}

void ivmthread::setRoot(const int* perm){
    ivmbb->setRoot(perm);
}


int
ivmthread::shareWork(std::shared_ptr<sequentialbb<int>> thief_thread)
{
    int numShared = 0;
    int l         = 0;

    std::shared_ptr<ivm> thief(thief_thread->IVM);
    std::shared_ptr<ivm> IVM(ivmbb->IVM);

    while (IVM->getPosition(l) == IVM->getEnd(l) && l < IVM->getDepth() && l < pbb->size - 4) l++;

    if (IVM->getPosition(l) < IVM->getEnd(l))
    {
        numShared++;
        for (int i = 0; i < l; i++) {
            thief->setPosition(i, IVM->getPosition(i));
            thief->setRow(i,IVM->getRowPtr(i));
            thief->setDirection(i, IVM->getDirection(i));
        }
        for (int i = 0; i < pbb->size; i++) thief->setEnd(i, IVM->getEnd(i));

        thief->setRow(l,IVM->getRowPtr(l));
        thief->setDirection(l, IVM->getDirection(l));
        thief->setPosition(l,IVM->cuttingPosition(l, 2));
        IVM->setEnd(l, thief->getPosition(l) - 1);

        // remaining levels : align thief left, victim right
        for (int i = l + 1; i < pbb->size; i++) thief->setPosition(i, 0);
        for (int i = l + 1; i < pbb->size; i++) IVM->setEnd(i, pbb->size - i - 1);

        thief->setDepth(l);
    }

    return numShared;
}
