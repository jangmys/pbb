/*
Nawaz-Enscore-Ham (NEH) heuristic
[https://www.sciencedirect.com/science/article/abs/pii/0305048383900889]

using

Taillard's acceleration
[https://www.sciencedirect.com/science/article/abs/pii/037722179090090X;]


Author : Jan GMYS (jan.gmys@univ-lille.fr)
*/
#include <limits.h>
#include <iostream>
#include <algorithm>

#include "fastNEH.h"
#include "fastinsertremove.h"

#include "../util.h"


void fastNEH::initialSort(std::vector<int>& perm){
    util::sort_by_key<int>(perm,m->sumPT);
}

//run NEH without sorting initial job-list
void fastNEH::runNEH(std::vector<int>& perm, int &cost){
    m->reset();

    //==================================================
    std::vector<int> p1(1,perm[0]);

    std::vector<int> joblist;
    std::copy(perm.begin()+1, perm.end(), std::back_inserter(joblist));

    for(auto& j : joblist){
        m->bestInsert(p1, j, cost);
    }
    //==================================================

    std::copy(p1.begin(),p1.end(),perm.begin());
}

//fill [0...N-1] permutation, sort and run NEH
void fastNEH::run(std::vector<int>& perm, int &cost)
{
    m->reset();

    perm.resize(nbJob);
    std::iota(perm.begin(),perm.end(),0);

    util::sort_by_key<int>(perm,m->sumPT);

    runNEH(perm,cost);
}

void fastNEH::run(std::shared_ptr<subproblem> p)
{
    m->reset();

    //fill 0,1,...N-1
    p->schedule.resize(nbJob);
    std::iota(p->schedule.begin(),p->schedule.end(),0);

    util::sort_by_key<int>(p->schedule,m->sumPT);

    int makespan;
    runNEH(p->schedule,makespan);
    p->ub = makespan;
};



subproblem fastNEH::operator()()
{
    subproblem perm(nbJob);

    int makespan;

    run(perm.schedule,makespan);
    perm.ub = makespan;

    return perm;
}
