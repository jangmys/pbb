#include <iostream>
#include <limits.h>
#include <string.h>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

#include "bound_fsp_weak_idle.h"

/*
In addition to LB1 this bound evaluates the weighted idle time as in

https://hal.archives-ouvertes.fr/hal-02937115/document

Luc Libralesso, Pablo Focke, Aurélien Secardin, Vincent Jost. Iterative beam search algorithms for
the permutation flowshop. 2020. ffhal-02937115f
*/


// lower bound and evaluation function for the PFSP
//
// single-machine based bound as in ...

// ==============================================================
// INITIALIZATIONS
// ==============================================================
void
bound_fsp_weak_idle::init(instance_abstract& _instance)
{
    pthread_mutex_lock(&_instance.mutex_instance_data);
    // get instance parameters (N/M)
    (_instance.data)->seekg(0);
    (_instance.data)->clear();
    *(_instance.data) >> nbJob;
    *(_instance.data) >> nbMachines;

    // read matrix of processing times from instance-data (stringstream)
    PTM = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));

    for (int j = 0; j < nbMachines; j++)
        for (int i = 0; i < nbJob; i++)
            *(_instance.data) >> PTM[j][i];
    pthread_mutex_unlock(&_instance.mutex_instance_data);

    // fill auxiliary data for LB computation
    fillMinHeadsTails();
}

void
bound_fsp_weak_idle::fillMinHeadsTails()
{
    min_heads = std::vector<int>(nbMachines);
    min_tails = std::vector<int>(nbMachines);

    // 1/ min start times on each machine
    for (int k = 0; k < nbMachines; k++) min_heads[k] = INT_MAX;
    // per definition =0 on first machine
    min_heads[0] = 0;

    for (int i = 0; i < nbJob; i++) {
        std::vector<int> tmp(nbMachines, 0);

        tmp[0] += PTM[0][i];
        for (int k = 1; k < nbMachines; k++) {
            tmp[k] = tmp[k - 1] + PTM[k][i];
        }
        for (int k = 1; k < nbMachines; k++) {
            min_heads[k] = (tmp[k - 1] < min_heads[k]) ? tmp[k - 1] : min_heads[k];
        }
    }

    // 2/ min run-out times on each machine
    std::fill(min_tails.begin(), min_tails.end(), INT_MAX);
    // per definition =0 on last machine
    min_tails[nbMachines - 1] = 0;

    for (int i = 0; i < nbJob; i++) {
        std::vector<int> tmp(nbMachines, 0);

        tmp[nbMachines - 1] += PTM[nbMachines - 1][i];
        for (int k = nbMachines - 2; k >= 0; k--) {
            tmp[k] = tmp[k + 1] + PTM[k][i];
        }
        for (int k = nbMachines - 2; k >= 0; k--) {
            min_tails[k] = std::min(tmp[k + 1], min_tails[k]);
        }
    }
}

// ==============================================================
// Lower bound computation
// ==============================================================
// set heads : partial scheduling of 'permut' up to position limit1 (included)
void
bound_fsp_weak_idle::scheduleFront(const int * permut, int limit1, std::vector<int>& front, std::vector<float>& idleFront)
{
    // no jobs scheduled in front
    if (limit1 == -1) {
        for (int i = 0; i < nbMachines; i++)
            front[i] = min_heads[i];
        return;
    }

    std::fill(front.begin(), front.end(), 0.0);
    idleFront[0]=0.0;

    // forward schedule
    for (int i = 0; i < limit1 + 1; i++) {
        int job = permut[i];
        front[0] += PTM[0][job];
        for (int j = 1; j < nbMachines; j++) {
            idleFront[j] += (float)std::max(front[j - 1] - front[j],0);
            front[j] = std::max(front[j - 1], front[j]) + PTM[j][job];
        }
    }
}

// set tails : reverse partial scheduling of 'permut' from end to position limit2
void
bound_fsp_weak_idle::scheduleBack(const int * permut, int limit2, std::vector<int>& back, std::vector<float>& idleBack)
{
    // no jobs in back
    if (limit2 == nbJob) {
        for (int i = 0; i < nbMachines; i++)
            back[i] = min_tails[i];
        return;
    }

    std::fill(back.begin(), back.end(), 0);

    idleBack[nbMachines-1]=0.0;

    // reverse schedule
    for (int k = nbJob - 1; k >= limit2; k--) {
        int job = permut[k];
        back[nbMachines - 1] += PTM[nbMachines - 1][job];
        for (int j = nbMachines - 2; j >= 0; j--) {
            idleBack[j] += (float)std::max(back[j + 1] - back[j],0);
            back[j] = std::max(back[j], back[j + 1]) + PTM[j][job];
        }
    }
}

// sum of processing time for unscheduled jobs
void
bound_fsp_weak_idle::sumUnscheduled(const int * permut, int limit1, int limit2, std::vector<int>& remain)
{
    std::fill(remain.begin(), remain.end(), 0);

    for (int k = limit1 + 1; k < limit2; k++) {
        int job = permut[k];
        for (int j = 0; j < nbMachines; j++) {
            remain[j] += PTM[j][job];// ptm[j * nbJob + job];
        }
    }
}

void
bound_fsp_weak_idle::computePartial(const int * schedule, int limit1, int limit2, std::vector<int>& front, std::vector<float>& idleFront, std::vector<int>& back, std::vector<float>& idleBack, std::vector<int>& remain)
{
    scheduleFront(schedule, limit1, front, idleFront);// set front[]
    scheduleBack(schedule, limit2, back, idleBack);// set back[]
    sumUnscheduled(schedule, limit1, limit2, remain);// set remain[]
}

// adds job to partial schedule in front and computes lower bound on optimal cost
// NB1: schedule is no longer needed at this point
// NB2: front, remain and back need to be set before calling this
// NB3: also compute total idle time added to partial schedule (can be used a criterion for job ordering)
// nOps : m*(3 add+2 max)  ---> O(m)
int
bound_fsp_weak_idle::addFrontAndBound(int job, float &idle, const std::vector<int>& front, const std::vector<float>& idleFront, const std::vector<int>& back, const std::vector<float>& idleBack, const std::vector<int>& remain)
{
    int tmp1;
    int tmp0   = front[0] + PTM[0][job];
    int max_lb = front[0] + remain[0] + back[0]; // LB for machine 0

    idle = idleBack[0]/back[0];

    for (int j = 1; j < nbMachines; j++) {
        float idleM = idleFront[j] + (float)std::max(0,tmp0-front[j]);

        tmp1 = std::max(front[j], tmp0);// job starts at tmp1 on machine j
        int lb = tmp1 + remain[j] + back[j];// LB for machine j
        tmp0 = tmp1 + PTM[j][job];// update tmp0

        idle += idleM/tmp0;
        if(j!=nbMachines-1)
            idle += idleBack[j]/back[j];

        if (lb > max_lb) {
            max_lb = lb;
        }
    }

    return max_lb;
}

// ... same for back
int
bound_fsp_weak_idle::addBackAndBound(int job, float &idle, const std::vector<int>& front, const std::vector<float>& idleFront, const std::vector<int>& back, const std::vector<float>& idleBack, const std::vector<int>& remain)
{
    int tmp1;
    int tmp0   = back[(nbMachines - 1)] + PTM[nbMachines - 1][job];
    int max_lb = front[nbMachines - 1] + remain[nbMachines - 1] + back[(nbMachines - 1)];

    idle = idleFront[nbMachines-1]/front[nbMachines-1];

    for (int j = nbMachines - 2; j >= 0; j--) {
        float idleM = idleBack[j] + std::max(0,tmp0-back[j]);
        // idle += std::max(0, tmp0 - back[j]);

        tmp1 = std::max(tmp0, back[j]);
        int lb = front[j] + remain[j] + tmp1;
        tmp0 = tmp1 + PTM[j][job];

        idle += idleM/tmp0;
        if(j!=0)
            idle += idleFront[j]/front[j];

        if (lb > max_lb) {
            max_lb = lb;
        }
    }

    return max_lb;
}

// get all lower bounds for all children
// begin/end if both LB pointers are given
void
bound_fsp_weak_idle::boundChildren(std::vector<int> schedule, int limit1, int limit2, int * costsBegin, int * costsEnd, float * prioBegin, float * prioEnd)
{
    std::vector<int> front(nbMachines, 0);
    std::vector<int> back(nbMachines, 0);
    std::vector<int> remain(nbMachines, 0);
    std::vector<float> idleFront(nbMachines, 0.0f);
    std::vector<float> idleBack(nbMachines, 0.0f);

    if (costsBegin && costsEnd) {
        // BEGIN/END LOWER BOUNDS
        memset(costsBegin, 0, nbJob * sizeof(int));
        memset(costsEnd, 0, nbJob * sizeof(int));

        computePartial(schedule.data(), limit1, limit2, front, idleFront, back, idleBack, remain);
        for (int i = limit1 + 1; i < limit2; i++) {
            int job = schedule[i];
            costsBegin[job] = addFrontAndBound(job, prioBegin[job], front, idleFront, back, idleBack, remain);
            costsEnd[job]   = addBackAndBound(job, prioEnd[job], front, idleFront, back, idleBack, remain);

            if(limit1 == -1)
                prioBegin[job]=0;
            if(limit2 == nbJob)
                prioEnd[job]=0;
        }
    } else if (costsBegin) {
        // BEGIN
        memset(costsBegin, 0, nbJob * sizeof(int));

        computePartial(schedule.data(), limit1, limit2, front, idleFront, back, idleBack, remain);
        for (int i = limit1 + 1; i < limit2; i++) {
            int job = schedule[i];
            costsBegin[job] = addFrontAndBound(job, prioBegin[job], front, idleFront, back, idleBack, remain);
        }
    } else if (costsEnd) {
        // END
        memset(costsEnd, 0, nbJob * sizeof(int));

        computePartial(schedule.data(), limit1, limit2, front, idleFront, back, idleBack, remain);
        for (int i = limit1 + 1; i < limit2; i++) {
            int job = schedule[i];
            costsBegin[job] = addBackAndBound(job, prioEnd[job], front, idleFront, back, idleBack, remain);
        }
    }
}

// evaluate full permutation
int
bound_fsp_weak_idle::evalSolution(std::vector<int> permut)
{
    std::vector<int> tmp(nbMachines, 0);

    for (int i = 0; i < nbJob; i++) {
        int job = permut[i];
        tmp[0] += PTM[0][job];
        for (int j = 1; j < nbMachines; j++) {
            tmp[j] = std::max(tmp[j - 1], tmp[j]) + PTM[j][job];
        }
    }

    return tmp[nbMachines - 1];
}

/////////////////////////////////////////////
// get lower bound for one subproblem (...)
void
bound_fsp_weak_idle::bornes_calculer(std::vector<int> permutation, int limite1, int limite2, int * couts, int best)
{
    std::vector<int> front(nbMachines, 0);
    std::vector<int> back(nbMachines, 0);
    std::vector<int> remain(nbMachines, 0);
    std::vector<float> idleFront(nbMachines, 0.0f);
    std::vector<float> idleBack(nbMachines, 0.0f);

    scheduleFront(permutation.data(), limite1, front, idleFront);
    scheduleBack(permutation.data(), limite2, back, idleBack);
    sumUnscheduled(permutation.data(), limite1, limite2, remain);

    int lb;
    int tmp0, tmp1, cmax;

    tmp0 = front[0] + remain[0];
    lb   = tmp0 + back[0];

    for (int j = 1; j < nbMachines; j++) {
        tmp1 = front[j] + remain[j];
        tmp1 = std::max(tmp1, tmp0);
        cmax = tmp1 + back[j];
        lb   = std::max(lb, cmax);
        tmp0 = tmp1;
    }

    couts[0] = lb;
}

void
bound_fsp_weak_idle::bornes_calculer(std::vector<int> permutation, int limite1, int limite2)
{ }
