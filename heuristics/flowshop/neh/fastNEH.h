#ifndef FASTNEH_H_
#define FASTNEH_H_

#include <array>

#include "libbounds.h"
#include "fastinsertremove.h"

class fastNEH{
public:
    fastInsertRemove<int>* m;

    fastNEH(instance_abstract*_inst);
    ~fastNEH();

    void initialSort(std::vector<int>& perm);
    void runNEH(std::vector<int>& perm, int &cost);
private:
    int nbJob;
    // instance_abstract * instance;
};


#endif
