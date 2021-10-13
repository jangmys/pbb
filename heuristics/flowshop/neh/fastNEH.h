#ifndef FASTNEH_H_
#define FASTNEH_H_

#include <array>

#include "libbounds.h"

#include "fastinsertremove.h"

class fastNEH{
public:
    instance_abstract * instance;

    fastInsertRemove<int>* m;

    int nbJob;

    fastNEH(instance_abstract*_inst);
    ~fastNEH();

    void initialSort(std::vector<int>& perm);
    void runNEH(std::vector<int>& perm, int &cost);
};


#endif
