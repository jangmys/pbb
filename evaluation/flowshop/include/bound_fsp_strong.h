#ifndef BOUND_FSP_STRONG_H
#define BOUND_FSP_STRONG_H

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <vector>
#include <tuple>

#include "bound_abstract.h"

class pbab;

class bound_fsp_strong : public bound_abstract<int> {
public:
    int nbJob;
    int nbMachines;
    int nbMachinePairs;

    int          branchingMode;
    int          earlyExit;
    int          machinePairs;

    // this is CONSTANT data. in multi-core BB each thread will instantiate the lower bound. making the following static will save some space ("shared"), but performance hits are observed especially on dual-socket NUMA nodes.
    std::vector<std::vector<int>> PTM;
    // for each machine k, minimum time between t=0 and start of any job
    std::vector<int> min_heads;
    // for each machine k, minimum time between release of any job and end of processing on the last machine
    std::vector<int> min_tails;

    std::vector<std::pair<int,int>> machine_pairs;

    void fillMinHeadsTails();
    void fillLags();
    void fillMachinePairs();
    void fillJohnsonSchedules();

    std::vector<std::vector<int>> johnson_schedules;
    std::vector<std::vector<int>> lags;

    std::vector<int> flag;

    int *        rewards;
    int *        countMachinePairs;
    int *        machinePairOrder;
    // int *        pluspetit[2];

    void
    init(instance_abstract * _instance);
    void
    configureBound(const int, const int, const int);

    void initCmax(std::pair<int,int>& tmp, std::pair<int,int>& ma, int ind);
    void cmaxFin(std::pair<int,int>& tmp, std::pair<int,int> ma);
    void heuristiqueCmax(std::pair<int,int>& tmp, std::pair<int,int> ma, int ind);
    int
    borneInfMakespan(int * valBorneInf, int minCmax);

    int
    borneInfLearn(int * valBorneInf, int UB, bool earlyExit);


    void
    machineBound(int * cost);
    int
    calculBorne(int minCmax);
    int
    johnsonUB(int permutation[], int limit2, int ind);

    int nbbounds;

    void
    scheduleFront(int permutation[], int limite1, int limite2, int * idle);

    void
    setFlags(int permutation[], int limite1, int limite2);
    //
    void
    scheduleBack(int permutation[], int limite2, int * idle);

    void
    boundChildren(int permutation[], int limite1, int limite2, int * costsBegin, int * costsEnd, int * prioBegin, int * prioEnd, int best);

    void
    bornes_calculer(int permutation[], int limite1, int limite2, int * couts, int);
    int
    bornes_calculer(int permutation[], int limite1, int limite2);

    int
    evalSolution(int * permut);

    void
    partial_cost(int permutation[], int limit1, int limit2, int * couts, int jobin, int here);

    ~bound_fsp_strong(){ };
private:
    std::vector<int> front;
    std::vector<int> back;
    std::vector<int> remain;
};

#endif // ifndef BOUND_FSP_STRONG_H
