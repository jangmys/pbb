// ==========================================================================================
#include <pthread.h>
#include <semaphore.h>
#include <sys/sysinfo.h>

#include <memory>

#include "libbounds.h"
#include "libheuristic.h"

#include "pbab.h"
#include "log.h"


pbab::pbab() : inst(std::make_shared<instance_dummy>("8")),size(inst->size),best_found(size),stats()
{
    this->ttm = new ttime();
}

pbab::pbab(std::shared_ptr<instance_abstract> _inst) : inst(_inst),size(inst->size),best_found(size),stats()
{
    this->ttm = new ttime();
}

void pbab::set_initial_solution(const std::vector<int> perm, const int cost)
{
    for(int i=0; i<size; i++){
        best_found.initial_perm[i] = perm[i];
        best_found.perm[i] = perm[i];
    }
    best_found.initial_cost.store(cost);
    best_found.cost.store(cost);
}

void pbab::set_initial_solution()
{
    int cost=INT_MAX;

    std::vector<int>perm(size);
    for(int i=0; i<size; i++){
        perm[i]=i;
    }

    //by default initial upper bound is INFTY
    best_found.cost.store(INT_MAX);

    switch (arguments::problem[0]) {
        case 'f':
        {
            switch (arguments::init_mode) {
                case -1: //initial ub passed as value
                {
                    set_initial_solution(perm,arguments::initial_ub);
                    std::cout<<"\t#Initial upper bound : \t"<<arguments::initial_ub<<"\n";
                    break;
                }
                case 0: // get from file
                {
                    switch (arguments::inst_name[0]) {
                        case 't': //Taillard
                        {
                            cost = std::static_pointer_cast<instance_taillard>(inst)->read_initial_ub_from_file(arguments::inst_name);
                            break;
                        }
                        case 'V': //VFR
                        {
                            cost = std::static_pointer_cast<instance_vrf>(inst)->get_initial_ub_from_file(arguments::inst_name);
                            break;
                        }
                    }
                    set_initial_solution(perm,cost);

                    std::cout<<"\t#Initial upper bound - FILE : \t"<<cost<<"\n";
                    break;
                }
                case 1:
                {
                    fastNEH neh(*(inst.get()));

                    neh.initialSort(perm);
                    neh.runNEH(perm,cost);

                    set_initial_solution(perm,cost);

                    std::cout<<"\t#Initial upper bound - NEH : \t"<<cost<<"\n";
                    break;
                }
                case 2:
                {
                    Beam bs(this,*(inst.get()));
                    // Beam bs(this,instance.get());

                    std::shared_ptr<subproblem> p = std::make_shared<subproblem>(size);

                    bs.run_loop(1<<14,p.get());

                    set_initial_solution(bs.bestSolution->schedule,p->fitness());

                    break;
                }
                case 3:
                {
                    break;
                }
                default:
                {
                    best_found.cost.store(INT_MAX);
                }
            }
            break;
        }
        case 'd':
        {
            break;
        }
    }


}


void
pbab::printStats()
{
    if(arguments::singleNode){
        stats.print();
    }else{
        ttm->printElapsed(ttm->wall,"TotalElapsed\t");
        ttm->printElapsed(ttm->masterWalltime,"MasterWalltime\t");
        stats.print();
    }

    if(best_found.foundAtLeastOneSolution)
    {
        std::cout<<"Found optimal solution:\n"<<std::endl;
        std::cout<<"Min Cmax:\t"<<best_found.cost<<std::endl;
        std::cout<<"Argmin Cmax:\t"; best_found.print();
    }else{
        std::cout<<"Not improved..."<<std::endl;
        std::cout<<"Optimal makespan is >= "<<best_found.initial_cost<<" (initial solution) "<<std::endl;
    }
}

pbab::~pbab()
{
    // delete instance;
}
