#ifndef POOLBB_H
#define POOLBB_H

#include <memory>

#include <subproblem.h>
#include <solution.h>

#include "pbab.h"
#include "pool.h"

#include "libbounds.h"

class Poolbb{
public:
    Poolbb(pbab* _pbb);

    void set_root(subproblem& node);
    void set_root(solution& node);

    void decompose(
        subproblem& n,
        std::vector<std::unique_ptr<subproblem>>& children
    );

    void run();

private:
    std::unique_ptr<Pool> pool;
    int pbsize;

    std::unique_ptr<Pruning> prune;
    std::unique_ptr<Branching> branch;
    std::unique_ptr<bound_abstract<int>> bound;

};



#endif
