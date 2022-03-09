#ifndef BOUND_DUMMY_H
#define BOUND_DUMMY_H


#include "bound_abstract.h"


class bound_dummy : public bound_abstract<int> {
public:
    void init(instance_abstract* _instance);

    void bornes_calculer(int permutation[], int limite1, int limite2, int* couts, int best){};

    void bornes_calculer(int permutation[], int limite1, int limite2){} ;

    // in : subproblem p
    // out : costsBegin / costsEnd
    // compute bounds of children nodes of subproblem p
    // goal: avoid redundant computation of parts that are common to children nodes
    void
    boundChildren(int * schedule, int limit1, int limit2, int * costsBegin, int * costsEnd, int * prioBegin,
      int * prioEnd){};

    int evalSolution(int * permut){
        return 0;
    };

    int size;
};

#endif
