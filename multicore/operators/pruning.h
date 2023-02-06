#ifndef PRUNING_H_
#define PRUNING_H_

#include "subproblem.h"
#include <memory>
#include <limits.h>

class subproblem;

class Pruning{
public:
    //minimization is default
    Pruning(int _ub = INT_MAX) : local_best(_ub)
    {};

    virtual bool operator()(const int lb){
        return lb >= local_best;
    };

    //returns true if subproblem can be eliminatedfor further exploration
    bool operator()(const subproblem * pr)
    {
        return operator()(pr->lower_bound());
    }

    int local_best;
};

//only search for better (smaller) solutions
//(find ONE global minimizer)
class keepSmaller : public Pruning
{
public:
    keepSmaller(int _ub = INT_MAX) : Pruning(_ub){};

    bool operator()(const int lb) override
    {
        return lb >= local_best;
    }
};

//find all minimizing solutions
//(find ALL global minimizers)
class keepEqualOrSmaller : public Pruning
{
public:
    keepEqualOrSmaller(int _ub = INT_MAX) : Pruning(_ub){};

    bool operator()(const int cost)
    {
        return cost > local_best;
    }
};

// Pruning make_prune();


#endif
