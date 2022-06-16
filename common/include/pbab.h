#ifndef PBAB_H
#define PBAB_H

#include <atomic>
#include <memory>

#include <pthread.h>
#include <semaphore.h>
#include "../include/misc.h"
#include "../include/arguments.h"
#include "../include/statistics.h"
#include "subproblem.h"
#include "solution.h"
#include "ttime.h"

#include "libbounds.h"

#include "operator_factory.h"

class instance_abstract;
class solution;
class ttime;


class pbab
{
public:
    pbab();
    pbab(std::unique_ptr<instance_abstract>& _inst);
    ~pbab();

    std::unique_ptr<instance_abstract> instance;
    int size;


    void
    set_instance(char problem[], char inst_name[]);

    std::unique_ptr<subproblem> best_solution;
    solution * sltn;
    solution * root_sltn;

    void set_initial_solution();

    std::atomic<bool> foundAtLeastOneSolution{false};
    std::atomic<bool> foundNewSolution{false};

    ttime * ttm;

    statistics stats;//(0,0,0,0);
    void printStats();

    // pthread_mutex_t mutex_instance;

    std::unique_ptr<BoundFactoryInterface<int>> bound_factory;
    template<typename T>
    void set_bound_factory(std::unique_ptr<BoundFactoryInterface<T>> b)
    {
        bound_factory = std::move(b);
    }

    std::unique_ptr<PruningFactoryInterface> pruning_factory;
    void set_pruning_factory(std::unique_ptr<PruningFactoryInterface> p)
    {
        pruning_factory = std::move(p);
    }

    std::unique_ptr<BranchingFactoryInterface> branching_factory;
    void set_branching_factory(std::unique_ptr<BranchingFactoryInterface> b)
    {
        branching_factory = std::move(b);
    }


    void
    buildInitialUB();

    int initialUB;
};

#endif // ifndef PBAB_H
