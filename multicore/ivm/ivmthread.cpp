#include "ivmthread.h"


int
ivmthread::shareWork(int numerator, int denominator, sequentialbb *thief_thread)
{
    int numShared = 0;
    int l         = 0;

    ivm* thief = thief_thread->IVM;
    ivm* IVM = ivmbb->IVM;

    while (IVM->posVect[l] == IVM->endVect[l] && l < IVM->line && l < pbb->size - 4) l++;

    if (IVM->posVect[l] < IVM->endVect[l])
    {
        numShared++;
        for (int i = 0; i < l; i++) {
            thief->posVect[i] = IVM->posVect[i];
            for (int j = 0; j < pbb->size; j++) thief->jobMat[i * pbb->size + j] = IVM->jobMat[i * pbb->size + j];
            thief->dirVect[i] = IVM->dirVect[i];
        }
        for (int i = 0; i < pbb->size; i++) thief->endVect[i] = IVM->endVect[i];
        for (int i = 0; i < pbb->size; i++) thief->jobMat[l * pbb->size + i] = IVM->jobMat[l * pbb->size + i];
        thief->dirVect[l] = IVM->dirVect[l];

        thief->posVect[l] = IVM->cuttingPosition(l, 2);
        IVM->endVect[l]   = thief->posVect[l] - 1;

        // remaining levels : align thief left, victim right
        for (int i = l + 1; i < pbb->size; i++) thief->posVect[i] = 0;
        for (int i = l + 1; i < pbb->size; i++) IVM->endVect[i] = pbb->size - i - 1;

        thief->line = l;
    }

    // IVM->displayVector(IVM->posVect);
    // IVM->displayVector(IVM->endVect);
    // IVM->displayVector(thief->posVect);
    // IVM->displayVector(thief->endVect);
    // std::cout<<"numShared: "<<numShared<<"\n";

    return numShared;
}
