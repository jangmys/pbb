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

pbab::pbab()
{
	stats.johnsonBounds = ATOMIC_VAR_INIT(0);
	stats.simpleBounds  = ATOMIC_VAR_INIT(0);
    stats.totDecomposed = ATOMIC_VAR_INIT(0);
    stats.leaves        = ATOMIC_VAR_INIT(0);

	this->ttm = new ttime();

    pthread_mutex_init(&mutex_instance, NULL);
    set_instance(arguments::problem, arguments::inst_name);
    size     = instance->size;

    std::cout<<"SIZE:\t"<<size<<std::endl;

	sltn = new solution(size);
	sltn->cost = arguments::initial_ub;

	root_sltn = new solution(size);
	root_sltn->cost = arguments::initial_ub;

	if (arguments::problem[0] == 'n')
	    sltn->cost = 0;
}

void
pbab::printStats()
{
    if(arguments::singleNode){
        printf("=================\n");
        printf("Exploration stats\n");
        printf("=================\n");

        // printf("\t######Total %d\n", M);
        std::cout << "TOTdecomposed:\t " << stats.totDecomposed << std::endl;
        // std::cout << "TOTjohnson LBs:\t " << stats.johnsonBounds << std::endl;
        // std::cout << "TOTsimple LBs:\t " << stats.simpleBounds << std::endl;
        // std::cout << "TOTleaves:\t " << stats.leaves << std::endl;

        // if (arguments::branchingMode > 0) {
        //     printf("AvgBranchingFactor:\t %f\n", (double) (stats.simpleBounds / 2) / stats.totDecomposed);
        //     printf("LB1-PruningRate:\t %f\n", 1.0 - (double) stats.totDecomposed / (stats.simpleBounds / 2));
        //
        //     if (arguments::boundMode == 2)
        //         printf("LB2-SuccessRate:\t %f\n",
        //           (double) (stats.johnsonBounds - stats.totDecomposed) / stats.totDecomposed);
        // }

		std::cout<<"\n";
        if(foundAtLeastOneSolution)
        {
            std::cout<<"Found optimal solution."<<std::endl;
			std::cout<<"Cmax\t"<<sltn->cost<<std::endl;
			std::cout<<*(sltn);
        }else{
        	std::cout<<"Not improved..."<<std::endl;
			std::cout<<"Cmax\t"<<sltn->cost<<std::endl;
        	std::cout<<"Optimal makespan is >= "<<sltn->cost<<" (initial solution) "<<std::endl;
        }
    }else{
        printf("shutting down\n");
    }
}

pbab::~pbab()
{
    delete instance;
}

std::unique_ptr<bound_abstract<int>> pbab::createBound(int nb)
{
    if(arguments::problem[0]=='f'){
        if(arguments::boundMode == 0){
            if(nb==0){
                auto bd = std::make_unique<bound_fsp_weak>( );

                bd->init(instance);
                return bd;
            }
            if(nb==1){
                return nullptr;
            }
        }
        if(arguments::boundMode == 1){
            if(nb==0){
                auto bd2 = std::make_unique<bound_fsp_strong>( );
                bd2->init(instance);
                bd2->earlyExit=arguments::earlyStopJohnson;
                bd2->machinePairs=arguments::johnsonPairs;
                return bd2;
            }
            if(nb==1){
                return nullptr;
            }
        }
        if(arguments::boundMode == 2){
            if(nb==0){
                auto bd = std::make_unique<bound_fsp_weak>();
                bd->init(instance);
                return bd;
            }
            if(nb==1){
                auto bd2 = std::make_unique<bound_fsp_strong>();
                bd2->init(instance);
                bd2->branchingMode=arguments::branchingMode;
                bd2->earlyExit=arguments::earlyStopJohnson;
                bd2->machinePairs=arguments::johnsonPairs;
                return bd2;
            }
        }
	}

	std::cout<<"CreateBound: unknown problem\n";
	return 0;
}


void pbab::reset()
{
    stats.totDecomposed = 0;
    stats.johnsonBounds = 0;
    stats.simpleBounds  = 0;
    stats.leaves        = 0;

    foundAtLeastOneSolution.store(false);
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
