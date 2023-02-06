// ==========================================================================================
#include <pthread.h>
#include <semaphore.h>
#include <sys/sysinfo.h>

#include <memory>

#include "libbounds.h"
#include "libheuristic.h"

#include "pbab.h"
#include "log.h"


pbab::pbab() : inst(instance_dummy("8")),size(inst.size),best_found(size),stats()
{
    this->ttm = new ttime();
}

pbab::pbab(instance_abstract _inst) : inst(_inst),size(inst.size),best_found(size),stats()
{
    this->ttm = new ttime();
}

void pbab::set_initial_solution(const int* perm, const int cost)
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
    //by default initial upper bound is INFTY
    best_found.cost.store(INT_MAX);

    switch (arguments::problem[0]) {
        case 'f':
        {
            //if set, read initial UB from file
            switch (arguments::init_mode) {
                case 0:
                {
                    std::cout<<"\t#Get initial upper bound : FILE\n";

                    std::vector<int>perm(size);
                    int cost=INT_MAX;

                    for(int i=0; i<size; i++){
                        perm[i]=i;
                    }

                    switch (arguments::inst_name[0]) {
                        case 't':
                        {
                            cost = static_cast<instance_taillard&>(inst).read_initial_ub_from_file(arguments::inst_name);

                            // cost = dynamic_cast<instance_taillard&>(inst).read_initial_ub_from_file(arguments::inst_name);
                            // cost = (static_cast<instance_taillard*>(instance.get()))->read_initial_ub_from_file(arguments::inst_name);
                            break;
                        }
                        case 'V':
                        {
                            cost = static_cast<instance_vrf&>(inst).get_initial_ub_from_file(arguments::inst_name);
                            // cost = (static_cast<instance_vrf*>(instance.get()))->get_initial_ub_from_file(arguments::inst_name);
                            break;
                        }
                    }
                    set_initial_solution(perm.data(),cost);

                    break;
                }
                case 1:
                {
                    std::cout<<"\t#Get initial upper bound : NEH\n";

                    fastNEH neh(inst);
                    // fastNEH neh(instance.get());

                    std::shared_ptr<subproblem> p = std::make_shared<subproblem>(size);
                    // std::shared_ptr<subproblem> p = std::make_shared<subproblem>(size);

                    int fitness;

                    neh.initialSort(p.get()->schedule);
                    neh.runNEH(p->schedule,fitness);

                    p->set_fitness(fitness);

                    set_initial_solution(p->schedule.data(),p->fitness());

                    break;
                }
                case 2:
                {
                    Beam bs(this,inst);
                    // Beam bs(this,instance.get());

                    std::shared_ptr<subproblem> p = std::make_shared<subproblem>(size);

                    bs.run_loop(1<<14,p.get());

                    set_initial_solution(bs.bestSolution->schedule.data(),p->fitness());

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
        std::cout<<"Cmax\t"<<best_found.cost<<std::endl;
        std::cout<<"Optimal makespan is >= "<<best_found.cost<<" (initial solution) "<<std::endl;
    }

}

pbab::~pbab()
{
    // delete instance;
}
