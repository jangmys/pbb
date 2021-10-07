#include "arguments.h"
#include "subproblem.h"
#include "solution.h"

#include "libbounds.h"
#include "../libheuristic.h"

#include <vector>
#include <algorithm>


int main(int argc, char* argv[])
{
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

    instance_abstract *instance;
    switch (arguments::inst_name[0]) {//DIFFERENT INSTANCES...
        case 't': {
            instance = new instance_taillard(arguments::inst_name);
            break;
        }
        case 'V': {
            instance = new instance_vrf(arguments::inst_name);
            break;
        }
        case 'r': {
            instance = new instance_random(arguments::inst_name);
            break;
        }
        case '.': {
            instance = new instance_filename(arguments::inst_name);
            break;
        }
    }


    //==========================================================================
    fastNEH neh(instance);

    std::vector<int>perm(instance->size);

    std::generate(perm.begin(), perm.end(), [n = 0] () mutable { return n++; });

    neh.initialSort(perm);

    for(auto &e : perm)
    {
        std::cout<<e<<" ";
    }
    std::cout<<std::endl;

    int cost;

    neh.runNEH(perm,cost);

    for(auto &e : perm)
    {
        std::cout<<e<<" ";
    }
    std::cout<<std::endl;
    std::cout<<cost<<std::endl;
    //==========================================================================

    //==========================================================================

    //==========================================================================
    IG ils(instance);

    subproblem *p = new subproblem(instance->size);

    int c = ils.runIG(p);

    p->print();

    std::cout<<c<<std::endl;


    LocalSearch ls(instance);

    cost = ls(perm,-1,perm.size());

    for(auto &e : perm)
    {
        std::cout<<e<<" ";
    }
    std::cout<<std::endl;
    std::cout<<cost<<std::endl;


    // Beam bs(instance);
    // subproblem *q = new subproblem(instance->size);
    // bs.run(1<<12,q);
    // bs.bestSolution->print();


    Treeheuristic th(instance);
    subproblem *q2 = new subproblem(instance->size);

    th.run(q2,99999);

    // Tree tr(instance,0,instance->size);
    //
    //
    // tr.beamRun(2<<12,q);


}
