#ifndef BEAM_H
#define BEAM_H

#include <memory>

#include "tree.h"
#include "branching.h"
#include "pruning.h"

class Tree;

class Beam{
public:
    Beam(instance_abstract* inst);

    std::unique_ptr<Tree> tr;
    std::unique_ptr<pruning> prune;
    std::unique_ptr<branching> branch;
    std::unique_ptr<bound_fsp_weak_idle> eval;

    std::unique_ptr<subproblem> bestSolution;


    int run(const int maxBeamWidth, subproblem* p);
    bool step(int beamWidth,int localBest);

    std::vector<subproblem*> decompose(subproblem& n);
};

#endif
