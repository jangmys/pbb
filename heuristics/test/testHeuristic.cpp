#include "arguments.h"
#include "subproblem.h"
#include "solution.h"

#include "libbounds.h"
#include "libheuristic.h"

#include <vector>
#include <algorithm>


int main(int argc, char* argv[])
{
    //==========================================================================
    if(argc != 4)
    {
        std::cout<<"Usage: -z p=fsp,i=ta20 <N> with N=...\n";

        std::cout<<"... 0 : NEH\n";
        std::cout<<"... 1 : ILS\n";
        std::cout<<"... 2 : LS\n";
        std::cout<<"... 3 : BEAM\n";
        std::cout<<"... 4 : DFS-LS\n";
    }

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

    std::cout<<argv[3]<<std::endl;

    //initial solution
    // int cost;
    // std::vector<int>perm(instance->size);
    // std::generate(perm.begin(), perm.end(), [n = 0] () mutable { return n++; });
    subproblem *p = new subproblem(instance->size);

    switch(atoi(argv[3]))
    {
        case 0:
        {
            fastNEH neh(instance);

            neh.initialSort(p->schedule);
            neh.runNEH(p->schedule,p->ub);

            std::cout<<" = NEH :\t";
            break;
        }
        case 1:
        {
            IG ils(instance);

            p->ub = ils.runIG(p);

            std::cout<<" = ILS :\t";
            break;
        }
        case 2:
        {
            LocalSearch ls(instance);

            p->ub = ls(p->schedule,-1,p->size);

            std::cout<<" = LS :\t";
            break;
        }
        case 3:
        {
            Beam bs(instance);

            // subproblem *q = new subproblem(instance->size);
            bs.run(1<<14,p);
            *p = *(bs.bestSolution);

            std::cout<<" = BEAM :\t";
            break;
        }
        case 4:
        {
            Treeheuristic th(instance);

            th.run(p,0);

            std::cout<<" = DFLS :\t";
            break;
        }
        case 5:
        {
            Beam bs(instance);
            Treeheuristic th(instance);

            bs.run(1<<14,p);
            *p = *(bs.bestSolution);

            th.run(p,p->ub);

            std::cout<<" = BS + DFLS :\t";

            break;
        }

    }

    for(auto &e : p->schedule)
    {
        std::cout<<e<<" ";
    }
    std::cout<<" === "<<p->ub<<std::endl;
}
