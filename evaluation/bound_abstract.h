#ifndef BOUND_ABSTRACT_H
#define BOUND_ABSTRACT_H

#include "instance_abstract.h"

template<typename T>
class bound_abstract {
public:
    ~bound_abstract(){};

    virtual void init(instance_abstract * _instance) = 0;

    // boundSubproblem
    virtual void
     bornes_calculer(int permutation[], int limite1, int limite2, T* couts, T best) = 0;
    virtual T
     bornes_calculer(int permutation[], int limite1, int limite2) = 0;

    // in : subproblem p
    // out : costsBegin / costsEnd
    // compute bounds of children nodes of subproblem p
    // goal: avoid redundant computation of parts that are common to children nodes
    virtual void
    boundChildren(int * schedule, int limit1, int limit2, T * costsBegin, T * costsEnd, T * prioBegin, T * prioEnd, T best) = 0;

    virtual int
     evalSolution(int * permut) = 0;
};

#endif // ifndef BOUND_ABSTRACT_H
