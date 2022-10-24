#ifndef BOUND_FSP_WEAK_H
#define BOUND_FSP_WEAK_H

#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#include "bound_abstract.h"





// class pbab;

//PFSP : fitness and data are integer...
class bound_fsp_weak : public bound_abstract<int> {
public:
    bound_fsp_weak(){};

    int nbJob;
    int nbMachines;

    // this is CONSTANT data. in multi-core BB each thread will instantiate the lower bound. making the following static will save some space ("shared"), but performance hits are observed especially on dual-socket NUMA nodes.
    std::vector<std::vector<int>> PTM;

    std::vector<int> p_times;
    // for each machine k, minimum time between t=0 and start of any job
    std::vector<int> min_heads;
    // for each machine k, minimum time between release of any job and end of processing on the last machine
    std::vector<int> min_tails;

    void
    init(instance_abstract * _instance);
    void
    fillMinHeadsTails();

    void
    computePartial(int * permut, int limit1, int limit2);
    int
    addFrontAndBound(int job, int &prio);
    int
    addBackAndBound(int job, int &prio);

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

    void
    scheduleFront(const int * permut, const int limit1);
    void
    scheduleBack(const int * permut, const int limit2);
    void
    sumUnscheduled(const int * permut, const int limit1, const int limit2);
};

#endif // ifndef BOUND_FSP_WEAK_H
