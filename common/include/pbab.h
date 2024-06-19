#ifndef PBAB_H
#define PBAB_H

#include <atomic>
#include <memory>
#include <pthread.h>

#include "misc.h"
#include "arguments.h"
#include "statistics.h"
#include "subproblem.h"
#include "incumbent.h"

#include "ttime.h"

#include "libbounds.h"


/*
- instance
- best/incumbent
- timers
- stats
*/
class pbab
{
public:
    pbab();
    pbab(std::shared_ptr<instance_abstract> _inst);
    ~pbab();

    std::shared_ptr<instance_abstract> inst;
    int size;

    Incumbent<int> best_found;

    void set_initial_solution();
    void set_initial_solution(const std::vector<int> permutation, const int cost);

    std::atomic<bool> workUpdateAvailable{false};

    ttime * ttm;

    statistics stats;//(0,0,0,0);
    void printStats();
};

#endif // ifndef PBAB_H
