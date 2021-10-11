// ==========================================================================================
#include <pthread.h>
#include <semaphore.h>
#include <sys/sysinfo.h>

#include <memory>

#include "../include/subproblem.h"
#include "../include/solution.h"
#include "../include/ttime.h"
#include "../include/pbab.h"
#include "../include/log.h"

pbab::pbab() : stats()
{
    pthread_mutex_init(&mutex_instance, NULL);

    //set instance and problem size
    set_instance(arguments::problem, arguments::inst_name);
    size     = instance->size;

    this->ttm = new ttime();
}

void pbab::set_instance(char problem[],char inst_name[])
{
    switch(problem[0])//DIFFERENT PROBLEMS...
    {
        case 'f': //FLOWSHOP
        {
            switch (inst_name[0]) {//DIFFERENT INSTANCES...
                case 't': {
                    instance = new instance_taillard(inst_name);
                    break;
                }
                case 'V': {
                    instance = new instance_vrf(inst_name);
                    break;
                }
                case 'r': {
                    instance = new instance_random(inst_name);
                    break;
                }
                case '.': {
                    instance = new instance_filename(inst_name);
                }
            }
            break;
        }
    }
}

void pbab::set_initial_solution()
{
    sltn = new solution(size);
    root_sltn = new solution(size);

    //by default initial upper bound in INFTY
    sltn->cost = INT_MAX;

    //if set, read initial UB from file
    switch (arguments::init_mode) {
        case 0:
        {
            if(arguments::init_mode == 0){
                std::cout<<"Get initial upper bound from file..."<<std::endl;
                switch (arguments::inst_name[0]) {
                    case 't':
                    {
                        std::cout<<"here\n";
                        sltn->cost = (static_cast<instance_taillard*>(instance))->get_initial_ub_from_file(arguments::inst_name,arguments::init_mode);
                        break;
                    }
                    case 'V':
                    {
                        sltn->cost = (static_cast<instance_vrf*>(instance))->get_initial_ub_from_file(arguments::inst_name,arguments::init_mode);
                        break;
                    }
                }
            }
            break;
        }
        case 1:
        {
            break;
        }
        case 2:
        {
            break;
        }
        default:
        {
            sltn->cost = INT_MAX;
        }
    }

    *(root_sltn) = *(sltn);

    FILE_LOG(logDEBUG) << "Initializing at optimum " << sltn->cost;
    FILE_LOG(logDEBUG) << "Guiding solution " << *(root_sltn);
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
        std::cout<<"Optimal-Cmax\t"<<sltn->cost<<std::endl;
        std::cout<<"Solution:\n"<<*sltn;
    }else{
        std::cout<<"Not improved..."<<std::endl;
        std::cout<<"Cmax\t"<<sltn->cost<<std::endl;
        std::cout<<"Optimal makespan is >= "<<sltn->cost<<" (initial solution) "<<std::endl;
    }

}

pbab::~pbab()
{
    delete instance;
}
