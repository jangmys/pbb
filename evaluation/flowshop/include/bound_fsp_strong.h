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

    enum lb2_variant lb2_type;

    int nbJob;
    int nbMachines;
    int nbMachinePairs;

    bound_data* data_lb1;
    johnson_bd_data* data_lb2;

    int          branchingMode;
    int          earlyExit;
    int          machinePairs;

    std::vector<machine_pair> machine_pairs;

    // void fillMinHeadsTails();
    // void fillMachinePairs();

    std::vector<int> rewards;

    // int *        machinePairOrder;

    void
    init(instance_abstract * _instance);
    void
    configureBound(const int, const int, const int);

    // void initCmax(std::pair<int,int>& tmp, machine_pair& ma, int ind);
    // void cmaxFin(std::pair<int,int>& tmp, machine_pair ma);
    // void heuristiqueCmax(int *flag, std::pair<int,int>& tmp, machine_pair ma, int ind);
    // int
    // borneInfMakespan(int * valBorneInf, int minCmax);

    int
    borneInfLearn(int * flags,  const int *const front, const int* const back, int UB, bool earlyExit);


    int nbbounds;

    void
    boundChildren(int permutation[], int limite1, int limite2, int * costsBegin, int * costsEnd, int * prioBegin, int * prioEnd, int best);

    void
    bornes_calculer(int permutation[], int limite1, int limite2, int * couts, int);
    int
    bornes_calculer(int permutation[], int limite1, int limite2);

    int
    evalSolution(int * permut);
};

#endif // ifndef BOUND_FSP_STRONG_H
