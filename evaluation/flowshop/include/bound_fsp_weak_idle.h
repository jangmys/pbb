#ifndef BOUND_FSP_WEAK_IDLE_H
#define BOUND_FSP_WEAK_IDLE_H

#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#include "bound_abstract.h"

class pbab;

//PFSP : fitness and data are integer...
class bound_fsp_weak_idle {
public:
    bound_fsp_weak_idle(){};

    int nbJob;
    int nbMachines;

    // this is CONSTANT data. in multi-core BB each thread will instantiate the lower bound. making the following static will save some space ("shared"), but performance hits are observed especially on dual-socket NUMA nodes.
    std::vector<std::vector<int>> PTM;
    // for each machine k, minimum time between t=0 and start of any job
    std::vector<int> min_heads;
    // for each machine k, minimum time between release of any job and end of processing on the last machine
    std::vector<int> min_tails;

    void
    init(instance_abstract& _instance);
    void
    fillMinHeadsTails();

    void
    computePartial(const int * permut, int limit1, int limit2, std::vector<int>& front, std::vector<float>& idleFront, std::vector<int>& back, std::vector<float>& idleBack, std::vector<int>& remain);
    int
    addFrontAndBound(int job, float &prio, const std::vector<int>& front, const std::vector<float>& idleFront, const std::vector<int>& back, const std::vector<float>& idleBack, const std::vector<int>& remain);
    int
    addBackAndBound(int job, float &prio, const std::vector<int>& front, const std::vector<float>& idleFront, const std::vector<int>& back, const std::vector<float>& idleBack, const std::vector<int>& remain);

    void
    boundChildren(const int * permut, int limit1, int limit2, int * costsBegin, int * costsEnd, float * prioBegin, float * prioEnd);

    int
    evalSolution(int * permut);

    void
    bornes_calculer(int permutation[], int limite1, int limite2, int * couts, int best);
    void
    bornes_calculer(int permutation[], int limite1, int limite2);

private:
    void
    scheduleFront(const int * permut, int limit1, std::vector<int>& front, std::vector<float>& idleFront);
    void
    scheduleBack(const int * permut, int limit2, std::vector<int>& back, std::vector<float>& idleBack);
    void
    sumUnscheduled(const int * permut, int limit1, int limit2, std::vector<int>& remain);
};

#endif
