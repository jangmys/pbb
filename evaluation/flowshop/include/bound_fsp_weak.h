#ifndef BOUND_FSP_WEAK_H
#define BOUND_FSP_WEAK_H

#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#include "bound_abstract.h"

#include "c_bound_simple.h"

//Basically, wrapping "c_bound_simple.h/.c" to make it a child class of bound_abstract.
//PFSP : fitness and data are integer...
class bound_fsp_weak : public bound_abstract<int> {
public:
    bound_fsp_weak(){};
    ~bound_fsp_weak(){
        free_bound_data(data);
    };

    void
    init(instance_abstract * _instance);

    void
    boundChildren(int * permut, int limit1, int limit2, int * costsBegin, int * costsEnd, int * prioBegin, int * prioEnd, int best);

    int
    evalSolution(int * permut);

    void
    bornes_calculer(int permutation[], int limite1, int limite2, int * couts, int best);
    int
    bornes_calculer(int permutation[], int limite1, int limite2);
private:
    // this is CONSTANT data. in multi-core BB each thread will instantiate the lower bound. making the following static will save some space ("shared"), but performance hits are observed especially on dual-socket NUMA nodes.
    bound_data *data;
};

#endif // ifndef BOUND_FSP_WEAK_H
