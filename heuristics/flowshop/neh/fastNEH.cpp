#include <limits.h>
#include <iostream>
#include <algorithm>

#include "fastNEH.h"
#include "fastinsertremove.h"

#include "../util.h"


void fastNEH::initialSort(std::vector<int>& perm){
    util::sort_by_key<int>(perm,m->sumPT);
}

void fastNEH::runNEH(std::vector<int>& perm, int &cost){
    m->reset();

    int c1=m->computeHeads(perm, 2);
    std::swap(perm[0],perm[1]);
    int c2=m->computeHeads(perm, 2);
    if(c1<c2)std::swap(perm[0],perm[1]);

    std::vector<int> permOut(nbJob);
    for(int k=0;k<nbJob-2;k++){
        permOut[k]=perm[2+k];
    }
    perm.erase(perm.begin()+2,perm.end());

    int len=2;
    for(int k=0;k<nbJob-2;k++){
		m->bestInsert(perm, len, permOut[k], cost);

        // std::cout<<std::setw(2)<<"k="<<k+2<<"\t";
        // for(auto &e : perm)
        // {
        //     std::cout<<std::setw(2)<<e<<" ";
        // }
        // std::cout<<"\n";
    }
}

void fastNEH::run(std::vector<int>& perm, int &cost)
{
    m->reset();

    perm.resize(nbJob);
    std::iota(perm.begin(),perm.end(),0);

    util::sort_by_key<int>(perm,m->sumPT);

    std::cout<<"sorted :";
    for(auto &e : perm)
    {
        std::cout<<e<<" ";
    }
    std::cout<<"\n";

    runNEH(perm,cost);

    std::cout<<"NEH out :";
    for(auto &e : perm)
    {
        std::cout<<e<<" ";
    }
    std::cout<<"\n";

}
