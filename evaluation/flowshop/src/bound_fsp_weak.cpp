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
bound_fsp_weak::init(instance_abstract& _instance)
{
    pthread_mutex_lock(&_instance.mutex_instance_data);
    // get instance parameters (N/M)
    int n,m;
    (_instance.data)->seekg(0);
    (_instance.data)->clear();
    *(_instance.data) >> n;
    *(_instance.data) >> m;

    data = new_bound_data(n,m);

    // read matrix of processing times from instance-data (stringstream)
    for (int j = 0; j < m; j++){
        for (int i = 0; i < n; i++){
            *(_instance.data) >> data->p_times[j*n + i];
        }
    }
    pthread_mutex_unlock(&_instance.mutex_instance_data);

    // fill auxiliary data for LB computation
    fill_min_heads_tails(data);
}

// ==============================================================
// Lower bound computation
// ==============================================================

// get all lower bounds for all children
// begin/end if both LB pointers are given
void
bound_fsp_weak::boundChildren(std::vector<int> schedule, int limit1, int limit2, int * costsBegin, int * costsEnd, int * prioBegin, int * prioEnd, int best)
{
    if (costsBegin && costsEnd) {
        lb1_children_bounds(data,schedule.data(),limit1,limit2,costsBegin,costsEnd,prioBegin,prioEnd,0);
    } else if (costsBegin) {
        lb1_children_bounds(data,schedule.data(),limit1,limit2,costsBegin,costsEnd,prioBegin,prioEnd,-1);
    } else if (costsEnd) {
        lb1_children_bounds(data,schedule.data(),limit1,limit2,costsBegin,costsEnd,prioBegin,prioEnd,1);
    }
}

// evaluate full permutation
int
bound_fsp_weak::evalSolution(std::vector<int> permut)
{
    return eval_solution(data,permut.data());
}

/////////////////////////////////////////////
// get lower bound for one subproblem (...)
void
bound_fsp_weak::bornes_calculer(std::vector<int> permutation, int limit1, int limit2, int * couts, int best)
{
    *couts = lb1_bound(data, permutation.data(), limit1, limit2);
}

int
bound_fsp_weak::bornes_calculer(std::vector<int> permutation, int limit1, int limit2)
{
    return lb1_bound(data, permutation.data(), limit1, limit2);
}
