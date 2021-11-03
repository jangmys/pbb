#ifndef TREEHEURISTIC_H
#define TREEHEURISTIC_H

#include <memory>

#include "tree.h"
#include "branching.h"
#include "pruning.h"
#include "beam.h"

class Tree;
class Beam;

class Treeheuristic{
public:
    Treeheuristic(instance_abstract* inst);

    std::unique_ptr<Tree> tr;
    std::unique_ptr<pruning> prune;
    std::unique_ptr<branching> branch;
    std::unique_ptr<bound_fsp_weak_idle> eval;

    std::unique_ptr<subproblem> bestSolution;


    std::unique_ptr<LocalSearch> ls;
    std::unique_ptr<IG> ig;
    std::unique_ptr<Beam> beam;

    int run(subproblem *s, int _ub);

    void exploreNeighborhood(subproblem* s, long long int cutoff);

    std::vector<subproblem*> decompose(subproblem& n);
    void insert(std::vector<subproblem *>&ns);
};

#endif
