// ==========================================================================================
#include <pthread.h>
#include <semaphore.h>
#include <sys/sysinfo.h>

#include <memory>

#include "libbounds.h"
#include "libheuristic.h"

// #include "solution.h"
#include "ttime.h"
#include "pbab.h"
#include "log.h"


pbab::pbab() : stats()
{
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);

    pthread_mutex_init(&mutex_instance, &attr);

    //set instance and problem size
    set_instance(arguments::problem, arguments::inst_name);
    size     = instance->size;

    this->ttm = new ttime();

    sltn = new solution(size);
    root_sltn = new solution(size);

    best_solution = std::make_unique<subproblem>(size);

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
    //by default initial upper bound in INFTY
    sltn->cost = INT_MAX;

    struct timespec t1,t2;
    clock_gettime(CLOCK_MONOTONIC,&t1);

    //if set, read initial UB from file
    switch (arguments::init_mode) {
        case 0:
        {
            std::cout<<" === Get initial upper bound : FILE\n";
            switch (arguments::inst_name[0]) {
                case 't':
                {
                    sltn->cost = (static_cast<instance_taillard*>(instance))->read_initial_ub_from_file(arguments::inst_name);
                    break;
                }
                case 'V':
                {
                    sltn->cost = (static_cast<instance_vrf*>(instance))->get_initial_ub_from_file(arguments::inst_name);
                    break;
                }
            }
            break;
        }
        case 1:
        {
            std::cout<<" === Get initial upper bound : NEH\n";

            fastNEH neh(instance);

            std::shared_ptr<subproblem> p = std::make_shared<subproblem>(instance->size);

            int fitness;

            neh.initialSort(p.get()->schedule);
            neh.runNEH(p->schedule,fitness);

            p->set_fitness(fitness);

            for(int i=0; i<instance->size; i++){
                sltn->perm[i] = p->schedule[i];
            }
            sltn->cost = p->fitness();

            break;
        }
        case 2:
        {
            Beam bs(instance);

            std::shared_ptr<subproblem> p = std::make_shared<subproblem>(instance->size);

            bs.run_loop(1<<14,p.get());

            for(int i=0; i<instance->size; i++){
                sltn->perm[i] = bs.bestSolution->schedule[i];
            }
            sltn->cost = p->fitness();

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

    clock_gettime(CLOCK_MONOTONIC,&t2);
    std::cout<<"\tTime(InitialSolution):\t"<<(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9<<std::endl;


    *(root_sltn) = *(sltn);

    // FILE_LOG(logDEBUG) << "Initializing at optimum " << sltn->cost;
    // FILE_LOG(logDEBUG) << "Guiding solution " << *(root_sltn);
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
    delete instance;
}
