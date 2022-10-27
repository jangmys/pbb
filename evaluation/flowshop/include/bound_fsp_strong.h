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

#include "c_bound_simple.h"
#include "c_bound_johnson.h"

class pbab;

typedef std::pair<int,int> machine_pair;

class bound_fsp_strong : public bound_abstract<int> {
public:
    bound_fsp_strong(){};

    ~bound_fsp_strong(){
        free_bound_data(data_lb1);
        free_johnson_bd_data(data_lb2);
    };

    int nbJob;
    int nbMachines;
    int nbMachinePairs;

    bound_data* data_lb1;
    johnson_bd_data* data_lb2;

    int          branchingMode;
    int          earlyExit;
    int          machinePairs;

    // this is CONSTANT data. in multi-core BB each thread will instantiate the lower bound. making the following static will save some space ("shared"), but performance hits are observed especially on dual-socket NUMA nodes.
    std::vector<std::vector<int>> PTM;
    std::vector<int>p_times;

    std::vector<machine_pair> machine_pairs;

    void fillMinHeadsTails();
    void fillLags();
    void fillMachinePairs();
    void fillJohnsonSchedules();

    std::vector<std::vector<int>> johnson_schedules;
    std::vector<std::vector<int>> lags;
    std::vector<int> p_lags;

    std::vector<int> flag;

    int *        rewards;
    int *        countMachinePairs;
    int *        machinePairOrder;
    // int *        pluspetit[2];

    void
    init(instance_abstract * _instance);
    void
    configureBound(const int, const int, const int);

    void initCmax(std::pair<int,int>& tmp, machine_pair& ma, int ind);
    void cmaxFin(std::pair<int,int>& tmp, machine_pair ma);
    void heuristiqueCmax(int *flag, std::pair<int,int>& tmp, machine_pair ma, int ind);
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
    setFlags(int permutation[], int limite1, int limite2, int* flag);
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
private:
    std::vector<int> front;
    std::vector<int> back;
    std::vector<int> remain;
};

#endif // ifndef BOUND_FSP_STRONG_H
