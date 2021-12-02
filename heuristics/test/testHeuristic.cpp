#include "arguments.h"
#include "subproblem.h"

#include "libbounds.h"
#include "libheuristic.h"

#include <memory>
#include <iostream>

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

    std::shared_ptr<instance_abstract> instance;

    switch (arguments::inst_name[0]) {//DIFFERENT INSTANCES...
        case 't': {
            instance = std::make_shared<instance_taillard>(arguments::inst_name);
            break;
        }
        case 'V': {
            instance = std::make_shared<instance_vrf>(arguments::inst_name);
            break;
        }
        case 'r': {
            instance = std::make_shared<instance_random>(arguments::inst_name);
            break;
        }
        case '.': {
            instance = std::make_shared<instance_filename>(arguments::inst_name);
            break;
        }
    }

    //initial solution
    // int cost;
    // std::vector<int>perm(instance->size);
    // std::generate(perm.begin(), perm.end(), [n = 0] () mutable { return n++; });
    std::shared_ptr<subproblem> p = std::make_shared<subproblem>(instance->size);

    std::cout<<argv[3]<<std::endl;

    struct timespec t1,t2;
    clock_gettime(CLOCK_MONOTONIC,&t1);

    switch(atoi(argv[3]))
    {
        case 0:
        {
            fastNEH neh(instance.get());

            neh.initialSort(p->schedule);
            neh.runNEH(p->schedule,p->ub);

            std::cout<<" = NEH :\t";
            break;
        }
        case 1:
        {
            IG ils(instance.get());

            p->ub = ils.runIG(p.get());

            std::cout<<" = ILS :\t";
            break;
        }
        case 2:
        {
            LocalSearch ls(instance.get());

            p->ub = ls(p->schedule,-1,p->size);

            std::cout<<" = LS :\t";
            break;
        }
        case 3:
        {
            Beam bs(instance.get());

            // // subproblem *q = new subproblem(instance->size);
            // bs.run(1<<14,p.get());
            // *p = *(bs.bestSolution);
            //
            // std::cout<<" = BEAM :\t";
            break;
        }
        case 4:
        {
            Beam bs(instance.get());

            // subproblem *q = new subproblem(instance->size);
            bs.run_loop(1<<14,p.get());
            *p = *(bs.bestSolution);

            std::cout<<" = BEAM :\t";
            break;
        }
        case 5:
        {
            Treeheuristic th(instance.get());

            th.run(p,0);
            //
            std::cout<<" = DFLS :\t";
            break;
        }

    }

    clock_gettime(CLOCK_MONOTONIC,&t2);
    std::cout<<
        (t2.tv_sec - t1.tv_sec) +
        (t2.tv_nsec - t1.tv_nsec)/1e9 << std::endl ;

    for(auto &e : p->schedule)
    {
        std::cout<<e<<" ";
    }
    std::cout<<" === "<<p->ub<<std::endl;
}
