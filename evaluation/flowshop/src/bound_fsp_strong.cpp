#include <limits.h>
#include <iostream>
#include <algorithm>
//
#include "c_taillard.h"
// #include "c_bound_simple.h"
// #include "c_bound_johnson.h"

#include "bound_fsp_strong.h"

// lower bound and evaluation function for the PFSP
//
// m-machine bound bound as in ... Lageweg et al


// ==============================================================
// INITIALIZATIONS
// ==============================================================
void
bound_fsp_strong::init(instance_abstract * _instance)
{
    pthread_mutex_lock(&_instance->mutex_instance_data);
    //read N,M from instance stream
    (_instance->data)->seekg(0);
    (_instance->data)->clear();
    *(_instance->data) >> nbJob;
    *(_instance->data) >> nbMachines;

    //allocate bound struct (for LB1)
    data_lb1 = new_bound_data(nbJob,nbMachines);

    for (int j = 0; j < nbMachines; j++){
        for (int i = 0; i < nbJob; i++){
            *(_instance->data) >> data_lb1->p_times[j*nbJob + i];
        }
    }
    pthread_mutex_unlock(&_instance->mutex_instance_data);

    //fill data_lb1
    fill_min_heads_tails(data_lb1);

    //until here like one-machine bound...
    lb2_type = LB2_FULL;
    //allocate bound struct (for 2-machine bound)
    data_lb2 = new_johnson_bd_data(data_lb1,lb2_type);

    if(lb2_type == LB2_FULL){
        fill_machine_pairs(data_lb2,lb2_type);
    }
    fill_lags(data_lb1,data_lb2);
    fill_johnson_schedules(data_lb1,data_lb2);

    rewards = std::vector<int>(data_lb2->nb_machine_pairs,0);

    machinePairs = 0;
}

void
bound_fsp_strong::configureBound(const int _branchingMode, const int _earlyExit, const int _machinePairs)
{
    branchingMode = _branchingMode;
    earlyExit     = _earlyExit;
    machinePairs  = _machinePairs;
}

// ==============================================================
// Lower bound computation
// ==============================================================
int
bound_fsp_strong::borneInfLearn(int *flags, const int *const front, const int* const back, int UB, bool earlyExit)
{
    const int N = data_lb1->nb_jobs;
    const int M = data_lb1->nb_machines;
    const int max_pairs = data_lb2->nb_machine_pairs;

    // reset periodically...
    if (nbbounds > 100 * 2 * N) {
        nbbounds = 0;
        for (int k = 0; k < max_pairs; k++) {
            rewards[k] = 0;
        }
    }

    int *order = data_lb2->machine_pair_order;
    int i = 1;
    int j = 2;
    while (i < max_pairs) {
        if (rewards[order[i - 1]] < rewards[order[i]]) {
            std::swap(order[i - 1], order[i]);
            if ((--i)) continue;
        }
        i = j++;
    }

    int maxLB = 0;
    int bestind = 0;

    // restrict to best nbMachines
    int nbPairs = nbMachines;
    // learn...
    int best = UB;
    if (nbbounds < 2 * data_lb1->nb_jobs){
        nbPairs = max_pairs;
        best = INT_MAX; //disable early exit
    }

    maxLB = lb_makespan_learn(data_lb1, data_lb2, flags, front, back, best, nbPairs, &bestind);

    nbbounds++;
    rewards[bestind]++;

    return maxLB;
} // bound_fsp_strong::borneInfLearn


// ==============================
// COMPUTE BOUND
// ==============================
void
bound_fsp_strong::bornes_calculer(int permutation[], int limite1, int limite2, int * couts, int best)
{
    if (limite2 - limite1 == 1) {
        // printf("this happens\n");
        couts[0] = evalSolution(permutation);
    } else {
        std::vector<int>front(data_lb1->nb_machines);
        std::vector<int>back(data_lb1->nb_machines);

        schedule_front(data_lb1, permutation,limite1,front.data());
        schedule_back(data_lb1, permutation,limite2,back.data());

        int *flags =new int[data_lb1->nb_jobs];
        set_flags(permutation, limite1, limite2, data_lb1->nb_jobs, flags);

        if (machinePairs == 3) {
            couts[0] = borneInfLearn(flags, front.data(), back.data(), best, true);
        } else {
            couts[0] = lb_makespan(data_lb1,data_lb2,flags,front.data(),back.data(),best);
        }

        delete[]flags;
    }
    couts[1] = 0;
}

void
bound_fsp_strong::boundChildren(int permutation[], int limit1, int limit2, int * costsBegin, int * costsEnd, int * prioBegin, int * prioEnd, int best)
{
    std::vector<int>costs(2,0);

    for (int i = limit1 + 1; i < limit2; i++) {
        int job = permutation[i];

        //front
        if(costsBegin){
            std::swap(permutation[limit1 + 1], permutation[i]);
            bornes_calculer(permutation, limit1 + 1, limit2, costs.data(), best);
            costsBegin[job] = costs[0];
            prioBegin[job]=costs[1];
            std::swap(permutation[limit1 + 1], permutation[i]);
        }
        //back
        if(costsEnd){
            std::swap(permutation[limit2 - 1], permutation[i]);
            bornes_calculer(permutation, limit1, limit2 - 1, costs.data(), best);
            costsEnd[job] = costs[0];
            prioEnd[job]=costs[1];
            std::swap(permutation[limit2 - 1], permutation[i]);
        }
    }
}

int
bound_fsp_strong::evalSolution(int * permut)
{
    return eval_solution(data_lb1,permut);
}

int
bound_fsp_strong::bornes_calculer(int * schedule, int limit1, int limit2)
{
    int costs[2];
    bornes_calculer(schedule, limit1, limit2,costs,INT_MAX);
    return costs[0];
}
