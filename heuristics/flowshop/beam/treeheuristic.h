#ifndef TREEHEURISTIC_H
#define TREEHEURISTIC_H

#include <memory>

#include "tree.h"
#include "branching.h"
#include "pruning.h"

#include "solution.h"

class Tree;

class Treeheuristic{
public:
    Treeheuristic(instance_abstract* inst);

    std::unique_ptr<Tree> tr;
    std::unique_ptr<pruning> prune;
    std::unique_ptr<branching> branch;
    std::unique_ptr<subproblem> bestSolution;
    std::vector<std::unique_ptr<bound_abstract<int>>> lb;

    std::unique_ptr<LocalSearch> ls;
    std::unique_ptr<IG> ig;

    int run(subproblem *s, int _ub);

    void exploreNeighborhood(subproblem* s);

    std::vector<subproblem*> decompose(subproblem& n);
    void insert(std::vector<subproblem *>&ns);

    // int run(const int maxBeamWidth, subproblem* p);
    // bool step(int beamWidth,int localBest);

    // std::vector<subproblem*> decompose(subproblem& n);
};

#endif
