#ifndef FASTNEH_H_
#define FASTNEH_H_

#include <array>

#include "libbounds.h"
#include "fastinsertremove.h"

#include "subproblem.h"

class fastNEH{
public:
    explicit fastNEH(instance_abstract&_inst) :
        m(std::make_unique<fastInsertRemove>(_inst)),
        nbJob(m->nbJob)
    {};

    fastNEH(const std::vector<std::vector<int>> ptm, const int N, const int M) :
        m(std::make_unique<fastInsertRemove>(ptm,N,M)),
        nbJob(N)
    {};

    void initialSort(std::vector<int>& perm);
    void runNEH(std::vector<int>& perm, int &cost);
    void run(std::vector<int>& perm, int &cost);

    subproblem operator()();
private:
    std::unique_ptr<fastInsertRemove> m;
    int nbJob;
};


#endif
