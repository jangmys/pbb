#include "arguments.h"
#include "subproblem.h"

#include "libbounds.h"
#include "libheuristic.h"

#include "flowshop/include/c_taillard.h"

#include <memory>
#include <iostream>


//============================
// test PFSP heuristics
// (Taillard instances)
//============================
int main(int argc, char* argv[])
{
    //==========================================================================
    // parse args
    //==================================
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

    if(arguments::inst_name[0]!='t' && arguments::inst_name[0]!='a'){
        std::cout<<"only ta instances (ta01-ta120)\n";
        return -1;
    }

    //=====================================
    //make INSTANCE : TAILLARD
    //====================================
    auto instance = pbb_instance::make_inst(arguments::problem, arguments::inst_name);

    int inst_id = atoi(arguments::inst_name+2);
    int N,M;

    N=taillard_get_nb_jobs(inst_id);
    M=taillard_get_nb_machines(inst_id);

    std::cout<<" nb jobs : "<<N<<"\t\t nb machines : "<<M<<std::endl;
    std::cout<<" ==================================="<<std::endl;

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
    std::shared_ptr<subproblem> p = std::make_shared<subproblem>(instance->size);

    pbab * pbb = new pbab();

    struct timespec t1,t2;
    clock_gettime(CLOCK_MONOTONIC,&t1);

    switch(atoi(argv[3]))
    {
        case 0: //NEH heuristic
        {
            std::cout<<" = NEH :\t";

            fastNEH neh(p_times,N,M);
            neh.run(p);
            break;
        }
        case 1:
        {
            std::cout<<" = ILS :\t";

            IG ils(p_times,N,M);

            ils.run(p);
            break;
        }
        case 2:
        {
            LocalSearch ls(p_times,N,M);
            // LocalSearch ls(*(instance.get()));

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
            Beam bs(pbb,*(instance.get()));

            // // subproblem *q = new subproblem(instance->size);
            bs.run(1<<14,p.get());
            *p = *(bs.bestSolution);
            //
            // std::cout<<" = BEAM :\t";
            break;
        }
        case 4:
        {
            Beam bs(pbb,*(instance.get()));

            // subproblem *q = new subproblem(instance->size);
            bs.run_loop(1<<14,p.get());
            *p = *(bs.bestSolution);

            std::cout<<" = BEAM :\t";
            break;
        }
        case 5:
        {
            Treeheuristic th(pbb,*(instance.get()));

            th.run(p,0);

            std::cout<<" = DFLS :\t";
            break;
        }
        case 6:
        {
            std::cout<<"vNNEH\n";

            vNNEH vneh(p_times,N,M);

            vneh.run_plus(p,N);

            // for(int i=1;i<=N-1;i*=2){
            //     vneh.run(p,i);
            //     std::cout<<" === CMAX: "<<p->fitness()<<std::endl;
            // }
            // vneh.run(p,N-1);
            // std::cout<<" === CMAX: "<<p->fitness()<<std::endl;


            break;
        }
        case 7:
        {

        }


    }

    clock_gettime(CLOCK_MONOTONIC,&t2);
    std::cout<<"Elapsed:\t"<<1000*(t2.tv_sec - t1.tv_sec) +
        (t2.tv_nsec - t1.tv_nsec)/1e6 << " ms"<<std::endl ;

    for(auto &e : p->schedule)
    {
        std::cout<<e<<" ";
    }
    std::cout<<" === CMAX: "<<p->fitness()<<std::endl;
}
