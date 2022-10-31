#ifndef PBAB_H
#define PBAB_H

#include <atomic>
#include <memory>
#include <pthread.h>

#include "../include/misc.h"
#include "../include/arguments.h"
#include "../include/statistics.h"
#include "subproblem.h"
#include "solution.h"
#include "ttime.h"

#include "libbounds.h"

#include "operator_factory.h"

/*
- instance
- bb operator factories
- best
- stats
*/
class pbab
{
public:
    pbab();
    ~pbab();

    std::unique_ptr<instance_abstract> instance;
    int size;

    void set_instance(std::unique_ptr<instance_abstract> _inst);

    solution * sltn;
    solution * root_sltn;

    void set_initial_solution();
    void set_initial_solution(const int* permutation, const int cost);


    std::atomic<bool> foundAtLeastOneSolution{false};
    std::atomic<bool> foundNewSolution{false};

    std::atomic<bool> workUpdateAvailable{false};


    ttime * ttm;

    statistics stats;//(0,0,0,0);
    void printStats();

    //------------------------bounding-------------------------
    std::unique_ptr<BoundFactoryBase> bound_factory;
    void set_bound_factory(std::unique_ptr<BoundFactoryBase> b){
        bound_factory = std::move(b);
    }


    //------------------------pruning-------------------------
    enum pruning_ops{
            prune_greater,prune_greater_equal,
            prune_less,prune_less_equal
        };

    void choose_pruning(int choice){
        switch (choice) {
            case prune_greater:
                pruning_factory = std::make_unique<PruneLargerFactory>();
                break;
            case prune_greater_equal:
                pruning_factory = std::make_unique<PruneLargerEqualFactory>();
                break;
            case prune_less:
                break;
            case prune_less_equal:
                break;
        }
    }
    std::unique_ptr<PruningFactoryInterface> pruning_factory;

    //------------------------branching-------------------------
    std::unique_ptr<BranchingFactoryInterface> branching_factory;
    void set_branching_factory(std::unique_ptr<BranchingFactoryInterface> b){
        branching_factory = std::move(b);
    }

    int initialUB;
};

#endif // ifndef PBAB_H
