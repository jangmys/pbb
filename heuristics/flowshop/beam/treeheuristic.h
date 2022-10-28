#ifndef TREEHEURISTIC_H
#define TREEHEURISTIC_H

#include "../../common/include/rand.hpp"

#include <memory>

#include "pbab.h"
#include "tree.h"
#include "branching.h"
#include "pruning.h"
#include "beam.h"

class Tree;
class Beam;

class Treeheuristic{
public:
    Treeheuristic(pbab* pbb, instance_abstract* inst);

    pbab *pbb;

    std::unique_ptr<Tree> tr;
    std::unique_ptr<Pruning> prune;
    std::unique_ptr<Branching> branch;
    std::unique_ptr<bound_fsp_weak_idle> eval;

    std::unique_ptr<subproblem> bestSolution;


    std::unique_ptr<LocalSearch> ls;
    std::unique_ptr<IG> ig;
    std::unique_ptr<Beam> beam;

    int run(std::shared_ptr<subproblem>& s, int _ub);

    void exploreNeighborhood(std::shared_ptr<subproblem> s, long long int cutoff);

    std::vector<std::shared_ptr<subproblem>> decompose(subproblem& n);
    void insert(std::vector<std::shared_ptr<subproblem>>&ns);
};

#endif
