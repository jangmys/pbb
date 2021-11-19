#ifndef PRUNING_H_
#define PRUNING_H_

#include "subproblem.h"
#include <memory>

class subproblem;

class pruning{
public:
    //minimization is default
    pruning() : local_best(INT_MAX)
    {};

    virtual bool operator()(const int lb) = 0;

    //returns true if subproblem can be eliminatedfor further exploration
    bool operator()(const subproblem * pr)
    {
        return operator()(pr->cost);
    }

    int local_best;
};

//only search for better (smaller) solutions
//(find ONE global minimizer)
class keepSmaller : public pruning
{
public:
    keepSmaller() : pruning(){};

    bool operator()(const int lb)
    {
        return lb >= local_best;
    }
};

//find all minimizing solutions
//(find ALL global minimizers)
class keepEqualOrSmaller : public pruning
{
public:
    keepEqualOrSmaller() : pruning(){};

    bool operator()(const int cost)
    {
        return cost > local_best;
    }
};


#endif
