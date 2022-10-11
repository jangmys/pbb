#ifndef BOUND_DUMMY_H
#define BOUND_DUMMY_H


#include "bound_abstract.h"


class bound_dummy : public bound_abstract<int> {
public:
    void init(instance_abstract* _instance);

    void bornes_calculer(int permutation[], int limite1, int limite2, int* couts, int best)
    {
        couts[0]=0;
        couts[1]=0;
    };

    int bornes_calculer(int permutation[], int limite1, int limite2){
        return 0;
    } ;

    // in : subproblem p
    // out : costsBegin / costsEnd
    // compute bounds of children nodes of subproblem p
    // goal: avoid redundant computation of parts that are common to children nodes
    void
    boundChildren(int * schedule, int limit1, int limit2, int * costsBegin, int * costsEnd, int * prioBegin, int * prioEnd, int best){};

    int evalSolution(int * permut){
        return 1;
    };

    int size;
};

#endif
