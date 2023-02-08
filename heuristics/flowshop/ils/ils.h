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

    int runIG(std::shared_ptr<subproblem> s);
    int runIG(subproblem* s, int l1, int l2);

    // void destruction(std::vector<int>& perm, std::vector<int>& permOut, int k);
    std::vector<int> destruction(std::vector<int>& perm, int k, int a, int b);
    // void construction(std::vector<int>& perm, std::vector<int>& permOut, int k);
    void construction(std::vector<int>& perm, std::vector<int>& permOut, int k,int a, int b);

    bool acceptance(int tempcost, int cost, float param);

    void perturbation(int *perm, int k, int a, int b);
};


#endif
