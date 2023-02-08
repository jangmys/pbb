#include "arguments.h"
#include "subproblem.h"

#include "libbounds.h"
#include "libheuristic.h"

#include "flowshop/include/c_taillard.h"

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

    instance_abstract instance = pbb_instance::make_inst(arguments::problem, arguments::inst_name);

    std::cout<<" === solving "<<arguments::problem<<" - instance "<<atoi(arguments::inst_name+2)<<std::endl;


    //INSTANCE
    int inst_id = atoi(arguments::inst_name+2);

    int N,M;
    N=taillard_get_nb_jobs(inst_id);
    M=taillard_get_nb_machines(inst_id);

    std::vector<int>ptm(N*M,0);
    taillard_get_processing_times(ptm.data(),inst_id);

    std::vector<std::vector<int>>p_times(M,std::vector<int>(N,0));
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            p_times[i][j] = ptm[i*N+j];
        }
    }


    //initial solution
    // int cost;
    // std::vector<int>perm(instance->size);
    // std::generate(perm.begin(), perm.end(), [n = 0] () mutable { return n++; });
    std::shared_ptr<subproblem> p = std::make_shared<subproblem>(instance.size);

    std::cout<<argv[3]<<std::endl;

    pbab * pbb = new pbab();

    struct timespec t1,t2;
    clock_gettime(CLOCK_MONOTONIC,&t1);

    switch(atoi(argv[3]))
    {
        case 0:
        {
            fastNEH neh(p_times,N,M);

            p = std::make_shared<subproblem>(neh());
            break;
        }
        case 1:
        {
            IG ils(instance);

            p->set_fitness(ils.runIG(p));

            std::cout<<" = ILS :\t";
            break;
        }
        case 2:
        {
            LocalSearch ls(instance);

            auto cost = ls.localSearchBRE(p->schedule);
            std::cout<<"COST-LS-BRE : "<<cost<<"\n";

            std::generate(p->schedule.begin(), p->schedule.end(), [n = 0] () mutable { return n++; });

            cost = ls.localSearchKI(p->schedule,10);
            std::cout<<"COST-LS-KI : "<<cost<<"\n";

            p->set_fitness(ls(p->schedule,-1,p->size));

            std::cout<<" = LS :\t";
            break;
        }
        case 3:
        {
            // pbab* pbb = new pbab();

            Beam bs(pbb,instance);

            // // subproblem *q = new subproblem(instance->size);
            bs.run(1<<14,p.get());
            *p = *(bs.bestSolution);
            //
            // std::cout<<" = BEAM :\t";
            break;
        }
        case 4:
        {
            // pbab* pbb = new pbab();
            Beam bs(pbb,instance);

            // subproblem *q = new subproblem(instance->size);
            bs.run_loop(1<<14,p.get());
            *p = *(bs.bestSolution);

            std::cout<<" = BEAM :\t";
            break;
        }
        case 5:
        {
            // pbab* pbb = new pbab();
            Treeheuristic th(pbb,instance);

            th.run(p,0);

            std::cout<<" = DFLS :\t";
            break;
        }

    }

    clock_gettime(CLOCK_MONOTONIC,&t2);
    std::cout<<
        1000*(t2.tv_sec - t1.tv_sec) +
        (t2.tv_nsec - t1.tv_nsec)/1e6 << " ms"<<std::endl ;

    for(auto &e : p->schedule)
    {
        std::cout<<e<<" ";
    }
    std::cout<<" === CMAX: "<<p->fitness()<<std::endl;
}
