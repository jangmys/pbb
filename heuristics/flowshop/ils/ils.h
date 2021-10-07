#ifndef IG_H_
#define IG_H_

#include <memory>

#include "../../common/include/misc.h"
#include "subproblem.h"


#include "fastneighborhood.h"
#include "fastNEH.h"
#include "ls.h"

// #include "crossover.h"

class IG{
public:
    int igiter;

    int nbJob;
    int nbMachines;

    // int *perm;
    int *removed;

    fastNEH *neh;
    fspnhood<int> * nhood;

    LocalSearch* ls;
    // std::unique_ptr<LocalSearch> ls;
    // crossover *xover;

    std::vector<int> visitOrder;
    // int *visitOrder;

    int destructStrength;
    float acceptanceParameter;
    float avgPT;

    IG(instance_abstract * inst);
    ~IG();

    int makespan(subproblem* s);
    void runIG();

    int runIG(subproblem* s);
    int runIG(subproblem* s, int l1, int l2);

    void shuffle(int *array, int n);

    void destruction(int *perm, int *permOut, int k);
    void construction(std::vector<int>& perm, std::vector<int>& permOut, int k);
    void construction(std::vector<int>& perm, int *permOut, int k,int a, int b);

    bool acceptance(int tempcost, int cost, float param);
    void destruction(int *perm, int *permOut, int k, int a, int b);

    void perturbation(int *perm, int k, int a, int b);

    int localSearch(int* const perm, int l1, int l2);
    int localSearchBRE(int *arr, int l1, int l2);
    int localSearchKI(int *arr,int kmax);
    int localSearchPartial(int *arr,const int len);

    // int ris(subproblem* curr,subproblem* guiding);
    // int vbih(subproblem* current, subproblem* guiding);
};


#endif
