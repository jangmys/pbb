#include <limits.h>
#include <iostream>
#include <algorithm>

#include "fastNEH.h"
#include "fastinsertremove.h"

template<typename key_type>
void sort_by_key(std::vector<int>& prmu, const std::vector<key_type>& key)
{
    std::sort(prmu.begin(),prmu.end(),
    [&](const int a,const int b)
    {
        return key[a] > key[b];
    }
    );
}

void fastNEH::initialSort(std::vector<int>& perm){
    std::sort(perm.begin(),perm.end(),
    [&](const int a,const int b)
    {
        return m->sumPT[a] > m->sumPT[b];
    }
    );
}

void fastNEH::runNEH(std::vector<int>& perm, int &cost){
    int c1=m->computeHeads(perm, 2);
    std::swap(perm[0],perm[1]);
    int c2=m->computeHeads(perm, 2);
    if(c1<c2)std::swap(perm[0],perm[1]);

    std::vector<int> permOut(nbJob);
    for(int k=0;k<nbJob-2;k++){
        permOut[k]=perm[2+k];
    }

    int len=2;
    for(int k=0;k<nbJob-2;k++){
		m->bestInsert(perm, len, permOut[k], cost);
    }
}
