#ifndef BOUND_FSP_WEAK_H
#define BOUND_FSP_WEAK_H

#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#include "bound_abstract.h"

#include "c_bound_simple.h"


// class pbab;

//PFSP : fitness and data are integer...
class bound_fsp_weak : public bound_abstract<int> {
public:
    bound_fsp_weak(){};
    ~bound_fsp_weak(){
        free_bound_data(data);
    };

    int nbJob;
    int nbMachines;

    bound_data *data;

    // this is CONSTANT data. in multi-core BB each thread will instantiate the lower bound. making the following static will save some space ("shared"), but performance hits are observed especially on dual-socket NUMA nodes.
    std::vector<std::vector<int>> PTM;

    void
    init(instance_abstract * _instance);
    void
    fillMinHeadsTails();

    void
    computePartial(int * permut, int limit1, int limit2);
    // int
    // addFrontAndBound(int job, int &prio);
    // int
    // addBackAndBound(int job, int &prio);

    void
    boundChildren(int * permut, int limit1, int limit2, int * costsBegin, int * costsEnd, int * prioBegin, int * prioEnd, int best);

    int
    evalSolution(int * permut);

    void
    bornes_calculer(int permutation[], int limite1, int limite2, int * couts, int best);
    int
    bornes_calculer(int permutation[], int limite1, int limite2);
private:
    //performance vs local + passing into functions ?
    std::vector<int> front;
    std::vector<int> back;
    std::vector<int> remain;
};

#endif // ifndef BOUND_FSP_WEAK_H
