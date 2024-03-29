#ifndef BOUND_ABSTRACT_H
#define BOUND_ABSTRACT_H

#include "instance_abstract.h"
#include <vector>

template<typename T>
class bound_abstract {
public:
    virtual ~bound_abstract() = default;

    virtual void init(instance_abstract& _instance) = 0;

    // boundSubproblem
    virtual void
     bornes_calculer(std::vector<int> permutation, int limite1, int limite2, T* couts, T best) = 0;
    virtual T
     bornes_calculer(std::vector<int> permutation, int limite1, int limite2) = 0;

    // in : subproblem p
    // out : costsBegin / costsEnd
    // compute bounds of children nodes of subproblem p
    // goal: avoid redundant computation of parts that are common to children nodes
    virtual void
    boundChildren(std::vector<int> schedule, int limit1, int limit2, T * costsBegin, T * costsEnd, T * prioBegin, T * prioEnd, T best) = 0;

    virtual int
     evalSolution(std::vector<int> permut) = 0;
};

#endif // ifndef BOUND_ABSTRACT_H
