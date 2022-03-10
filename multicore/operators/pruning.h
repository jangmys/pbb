#ifndef PRUNING_H_
#define PRUNING_H_

#include "subproblem.h"
#include <memory>

class subproblem;

class Pruning{
public:
    //minimization is default
    Pruning() : local_best(INT_MAX)
    {};

    virtual bool operator()(const int lb) = 0;

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
    keepSmaller() : Pruning(){};

    bool operator()(const int lb)
    {
        return lb >= local_best;
    }
};

//find all minimizing solutions
//(find ALL global minimizers)
class keepEqualOrSmaller : public Pruning
{
public:
    keepEqualOrSmaller() : Pruning(){};

    bool operator()(const int cost)
    {
        return cost > local_best;
    }
};


#endif
