/*
Iterated Greedy

similar to Ruiz, Stützle
[https://www.sciencedirect.com/science/article/abs/pii/S0377221705008507]
[https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=db3494f9b0e0286b96f768a317fcdc25dd536f84]


Author : Jan GMYS (jan.gmys@univ-lille.fr)
*/
#ifndef IG_H_
#define IG_H_

#include <memory>

#include "../../common/include/rand.hpp"
// #include "../../common/include/misc.h"
#include "subproblem.h"


#include "../neh/fastNEH.h"
#include "../localsearch/fastneighborhood.h"
#include "../localsearch/ls.h"

// #include "crossover.h"

class IG{
public:
    IG(const std::vector<std::vector<int>> p_times, const int N, const int M);

    IG(instance_abstract& inst);

    std::unique_ptr<fspnhood<int>> nhood;
    std::unique_ptr<LocalSearch> ls;

    int nbJob;
    int nbMachines;

    int igiter;

    std::vector<int> visitOrder;

    int destructStrength;
    float acceptanceParameter;
    float avgPT;

    void run(std::shared_ptr<subproblem> s);

    int runIG(std::shared_ptr<subproblem> s, const int niter);
    int runIG(subproblem* s, int l1, int l2, const int niter);

    // void destruction(std::vector<int>& perm, std::vector<int>& permOut, int k);

    std::vector<int> destruction(std::vector<int>& perm, int k);
    std::vector<int> destruction(std::vector<int>& perm, int k, int a, int b);

    // void construction(std::vector<int>& perm, std::vector<int>& permOut, int k);
    void construction(std::vector<int>& perm, std::vector<int>& permOut);
    void construction(std::vector<int>& perm, std::vector<int>& permOut, int k,int a, int b);

    bool acceptance(int tempcost, int cost, float param);

    // void perturbation(int *perm, int k, int a, int b);
};


#endif
