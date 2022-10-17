// ==========================================================================================
#include <pthread.h>
#include <semaphore.h>
#include <sys/sysinfo.h>

#include <memory>

#include "libbounds.h"
#include "libheuristic.h"

// #include "solution.h"
#include "pbab.h"
// #include "ttime.h"
#include "log.h"


pbab::pbab() : stats()
{
    //set instance and problem size
    // set_instance(arguments::problem, arguments::inst_name);
    //
    // size     = instance->size;

    this->ttm = new ttime();

    // sltn = new solution(size);
    // root_sltn = new solution(size);
    //
    // best_solution = std::make_unique<subproblem>(size);
}

// void pbab::set_instance(char problem[],char inst_name[])
// {
//     InstanceFactory inst_factory;
//
//     set_instance(inst_factory.make_instance(problem, inst_name));
// }

void
pbab::set_instance(std::unique_ptr<instance_abstract> _inst){
    instance = std::move(_inst);

    size     = instance->size;

    sltn = new solution(size);
    root_sltn = new solution(size);

    // best_solution = std::make_unique<subproblem>(size);
}


void pbab::set_initial_solution(const int* perm, const int cost)
{
    for(int i=0; i<size; i++){
        sltn->perm[i] = perm[i];
    }
    sltn->cost = cost;
    initialUB = cost;

    *(root_sltn) = *(sltn);
}

void pbab::set_initial_solution()
{
    //by default initial upper bound is INFTY
    sltn->cost = INT_MAX;

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
                            cost = (static_cast<instance_taillard*>(instance.get()))->read_initial_ub_from_file(arguments::inst_name);
                            break;
                        }
                        case 'V':
                        {
                            cost = (static_cast<instance_vrf*>(instance.get()))->get_initial_ub_from_file(arguments::inst_name);
                            break;
                        }
                    }
                    set_initial_solution(perm.data(),cost);

                    break;
                }
                case 1:
                {
                    std::cout<<"\t#Get initial upper bound : NEH\n";

                    fastNEH neh(instance.get());

                    std::shared_ptr<subproblem> p = std::make_shared<subproblem>(instance->size);

                    int fitness;

                    neh.initialSort(p.get()->schedule);
                    neh.runNEH(p->schedule,fitness);

                    p->set_fitness(fitness);

                    // for(int i=0; i<instance->size; i++){
                    //     sltn->perm[i] = p->schedule[i];
                    // }
                    // sltn->cost = p->fitness();

                    set_initial_solution(p->schedule.data(),p->fitness());

                    break;
                }
                case 2:
                {
                    Beam bs(this,instance.get());

                    std::shared_ptr<subproblem> p = std::make_shared<subproblem>(instance->size);

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
                    sltn->cost = INT_MAX;
                }
            }
            break;
        }
        case 'd':
        {
            set_initial_solution(sltn->perm,INT_MAX);
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

    if(foundAtLeastOneSolution)
    {
        std::cout<<"Found optimal solution:\n"<<std::endl;
        std::cout<<"Min Cmax:\t"<<sltn->cost<<std::endl;
        std::cout<<"Argmin Cmax:\t"; sltn->print();
    }else{
        std::cout<<"Not improved..."<<std::endl;
        std::cout<<"Cmax\t"<<sltn->cost<<std::endl;
        std::cout<<"Optimal makespan is >= "<<sltn->cost<<" (initial solution) "<<std::endl;
    }

}

pbab::~pbab()
{
    // delete instance;
}
