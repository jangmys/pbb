#ifndef BEAM_H
#define BEAM_H

#include <memory>

#include "pbab.h"
#include "tree.h"
#include "branching.h"
#include "pruning.h"

class Tree;

struct prio_compare {
    bool
    operator () (const std::unique_ptr<subproblem>& p1, const std::unique_ptr<subproblem>& p2) const
    {
        return p1->prio < p2->prio;
    }
};


class Beam{
public:
    Beam(pbab* _pbb,instance_abstract& inst);

    pbab* pbb;

    std::vector<std::unique_ptr<subproblem>>activeSet;

    std::unique_ptr<Tree> tr;
    std::unique_ptr<Pruning> prune;
    std::unique_ptr<Branching> branch;
    std::unique_ptr<bound_fsp_weak_idle> eval;

    std::unique_ptr<subproblem> bestSolution;


    int run(const int maxBeamWidth, subproblem* p);
    int run_loop(const int maxBeamWidth, subproblem* p);

    bool step(unsigned int beamWidth,int localBest);
    bool step_loop(unsigned int beamWidth,int localBest);
    bool step_loop_pq(unsigned int beamWidth,int localBest);
    bool step_loop_local_pq(unsigned int beamWidth,int localBest);

    void decompose(const subproblem& n, std::vector<std::unique_ptr<subproblem>>& ns);
    // std::vector<subproblem*> decompose(const subproblem& n);
};

#endif
