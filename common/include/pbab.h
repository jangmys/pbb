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

#include "libbounds.h"

# define ALIGN 64

class instance_abstract;
class solution;
class ttime;

class pbab
{
public:
    pbab();
    ~pbab();

    int size;
    instance_abstract* instance;

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

    pthread_mutex_t mutex_instance;

    void
    buildInitialUB();

    int initialUB;
};

#endif // ifndef PBAB_H
