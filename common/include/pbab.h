#ifndef PBAB_H
#define PBAB_H

#include <atomic>
#include <memory>

#include <pthread.h>
#include <semaphore.h>
#include "../include/misc.h"
#include "../include/arguments.h"
#include "../include/statistics.h"

#include "libbounds.h"


# define ALIGN 64

class instance_abstract;
// class bound_abstract<int>;
class solution;
class ttime;

class pbab
{
public:
    int size;

    instance_abstract * instance;

    std::unique_ptr<bound_abstract<int>> createBound(int nb);
    // bound_abstract<int>* createBound(int nb);

    void
    set_instance(char problem[], char inst_name[]);

    solution * sltn;
    solution * root_sltn;

    std::atomic<bool> foundAtLeastOneSolution{false};
    std::atomic<bool> foundNewSolution{false};

    ttime * ttm;

    statistics stats;//(0,0,0,0);
    void
    printStats();

    // void set_heuristic();

    pbab();
    ~pbab();

    void
    reset();

    pthread_mutex_t mutex_instance;

    // void
    // buildPriorityTables();
    void
    buildInitialUB();

    int initialUB;

    // std::vector<std::tuple<std::vector<int>, std::vector<int> > > remain;
};

#endif // ifndef PBAB_H
