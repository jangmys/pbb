#include <iostream>
#include <limits.h>
#include <string.h>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <stdlib.h>

#include "bound_fsp_weak.h"

#include "c_bound_simple.h"

// lower bound and evaluation function for the PFSP
//
// single-machine based bound as in ...

// ==============================================================
// INITIALIZATIONS
// ==============================================================
void
bound_fsp_weak::init(instance_abstract * _instance)
{
    pthread_mutex_lock(&_instance->mutex_instance_data);
    // get instance parameters (N/M)
    (_instance->data)->seekg(0);
    (_instance->data)->clear();
    *(_instance->data) >> nbJob;
    *(_instance->data) >> nbMachines;

    data = new_bound_data(nbJob,nbMachines);

    // read matrix of processing times from instance-data (stringstream)
    PTM = std::vector<std::vector<int> >(nbMachines, std::vector<int>(nbJob));

    // p_times = std::vector<int>(nbJob*nbMachines);

    for (int j = 0; j < nbMachines; j++){
        for (int i = 0; i < nbJob; i++){
            *(_instance->data) >> PTM[j][i];

            // p_times[j*nbJob + i] = PTM[j][i];
            data->p_times[j*nbJob + i] = PTM[j][i];
        }
    }
    pthread_mutex_unlock(&_instance->mutex_instance_data);

    // fill auxiliary data for LB computation
    fill_min_heads_tails(data,data->min_heads,data->min_tails);
    // fillMinHeadsTails();

    // tempory memory needed at each bound computation
    front  = std::vector<int>(nbMachines);
    back   = std::vector<int>(nbMachines);
    remain = std::vector<int>(nbMachines);

}

void
bound_fsp_weak::fillMinHeadsTails()
{
    fill_min_heads_tails(data,data->min_heads,data->min_tails);
}






// ==============================================================
// Lower bound computation
// ==============================================================
void
bound_fsp_weak::computePartial(int * schedule, int limit1, int limit2)
{
    schedule_front(data->p_times, data->min_heads, nbJob, nbMachines, schedule,limit1,front.data());
    // schedule_front(data, schedule,limit1,front.data()); // set front[]

    schedule_back(data->p_times, data->min_tails, nbJob, nbMachines, schedule,limit2,back.data());
    // schedule_back(data, schedule,limit2,back.data());   // set back[]
    sum_unscheduled(data, schedule, limit1, limit2, remain.data()); // set remain[]
}

// get all lower bounds for all children
// begin/end if both LB pointers are given
void
bound_fsp_weak::boundChildren(int * schedule, int limit1, int limit2, int * costsBegin, int * costsEnd, int * prioBegin, int * prioEnd, int best)
{
    if (costsBegin && costsEnd) {
        // BEGIN/END LOWER BOUNDS
        memset(costsBegin, 0, nbJob * sizeof(int));
        memset(costsEnd, 0, nbJob * sizeof(int));

        computePartial(schedule, limit1, limit2);
        for (int i = limit1 + 1; i < limit2; i++) {
            int job = schedule[i];

            costsBegin[job] = add_front_and_bound(data,job,front.data(),back.data(),remain.data(),&prioBegin[job]);
            costsEnd[job]   = add_back_and_bound(data,job,front.data(),back.data(),remain.data(),&prioEnd[job]);
        }
    } else if (costsBegin) {
        // BEGIN
        memset(costsBegin, 0, nbJob * sizeof(int));

        computePartial(schedule, limit1, limit2);
        for (int i = limit1 + 1; i < limit2; i++) {
            int job = schedule[i];
            costsBegin[job] = add_front_and_bound(data,job,front.data(),back.data(),remain.data(),&prioBegin[job]);
        }
    } else if (costsEnd) {
        // END
        memset(costsEnd, 0, nbJob * sizeof(int));

        computePartial(schedule, limit1, limit2);
        for (int i = limit1 + 1; i < limit2; i++) {
            int job = schedule[i];
            costsEnd[job] = add_back_and_bound(data,job,front.data(),back.data(),remain.data(),&prioEnd[job]);
            // addBackAndBound(job, prioEnd[job]);
        }
    }
}

// evaluate full permutation
int
bound_fsp_weak::evalSolution(int * permut)
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
bound_fsp_weak::bornes_calculer(int permutation[], int limit1, int limit2, int * couts, int best)
{
    computePartial(permutation, limit1, limit2);

    couts[0] = machine_bound_from_parts(front.data(),back.data(),remain.data(),nbMachines);
}

int
bound_fsp_weak::bornes_calculer(int permutation[], int limite1, int limite2)
{ return 0;}
