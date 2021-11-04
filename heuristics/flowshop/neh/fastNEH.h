#ifndef FASTNEH_H_
#define FASTNEH_H_

#include <array>

#include "libbounds.h"
#include "fastinsertremove.h"

class fastNEH{
public:
    explicit fastNEH(instance_abstract*_inst) :
        m(std::make_unique<fastInsertRemove>(_inst)),
        nbJob(m->nbJob)
    {};

    void initialSort(std::vector<int>& perm);
    void runNEH(std::vector<int>& perm, int &cost);
private:
    std::unique_ptr<fastInsertRemove> m;
    int nbJob;
    // instance_abstract * instance;
};


#endif
