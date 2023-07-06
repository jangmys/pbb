#ifndef VNNEH_H_
#define VNNEH_H_

#include <memory>
#include <iostream>
#include <cmath>

#include "subproblem.h"
#include "../neh/fastinsertremove.h"
#include "../neh/fastNEH.h"
#include "../util.h"

class vNNEH{
public:
    vNNEH(const std::vector<std::vector<int>> p_times, const int N, const int M) : nbJob(N), m(std::make_unique<fastInsertRemove>(p_times,N,M)), neh(std::make_unique<fastNEH>(p_times,N,M)){
        std::cout<<"vNNEH\n";
    }

    int nbJob;
    std::unique_ptr<fastInsertRemove> m;
    std::unique_ptr<fastNEH> neh;

    void run_me(std::shared_ptr<subproblem> p, const int N);

    void run(std::shared_ptr<subproblem> p, const int N);
    void run_plus(std::shared_ptr<subproblem> p, const int N);
};



#endif
