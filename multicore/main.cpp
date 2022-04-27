#include "arguments.h"
#include "pbab.h"
#include "solution.h"
#include "log.h"
#include "ttime.h"

#include "libbounds.h"
#include "matrix_controller.h"
#include "../ivm/sequentialbb.h"

// instance_abstract*
std::unique_ptr<instance_abstract>
set_instance(char problem[],char inst_name[])
{
    instance_abstract* instance;

    switch(problem[0])//DIFFERENT PROBLEMS...
    {
        case 'f': //FLOWSHOP
        {
            switch (inst_name[0]) {//DIFFERENT INSTANCES...
                case 't': {
                    return std::make_unique<instance_taillard>(inst_name);
                    // instance = new instance_taillard(inst_name);
                    break;
                }
                case 'V': {
                    return std::make_unique<instance_vrf>(inst_name);
                    // instance = new instance_vrf(inst_name);
                    break;
                }
                case 'r': {
                    return std::make_unique<instance_random>(inst_name);
                    // instance = new instance_random(inst_name);
                    break;
                }
                case '.': {
                    return std::make_unique<instance_filename>(inst_name);
                    // instance = new instance_filename(inst_name);
                }
            }
            break;
        }
        case 'd': //DUMMY
        {
            return std::make_unique<instance_dummy>(inst_name);
            // instance = new instance_dummy(inst_name);
        }
    }
    // return instance;
}

template class PFSPBoundFactory<int>;


int
main(int argc, char ** argv)
{
    //PARAMETER PARSING
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

    //SET UP LOGGING
    FILELog::ReportingLevel() = logERROR;
#ifndef NDEBUG
    FILELog::ReportingLevel() = logDEBUG;
#endif
    FILE* log_fd = fopen(arguments::logfile, "w" );
    Output2FILE::Stream() = log_fd;

    //SET INSTANCE
    std::unique_ptr<instance_abstract> inst = set_instance(arguments::problem, arguments::inst_name);


    pbab * pbb = new pbab(inst);

    // pbb->set_instance(arguments::problem, arguments::inst_name);

    //SET BOUND
    std::unique_ptr<BoundFactoryInterface<int>> bound;
    if(arguments::problem[0]=='f'){
        bound = std::make_unique<PFSPBoundFactory<int>>();
    }else if(arguments::problem[0]=='d'){
        std::cout<<"dummy\n";
    }
    pbb->set_bound_factory(bound);

    //SET PRUNING
    std::unique_ptr<PruningFactoryInterface> prune;
    if(arguments::findAll){
        prune = std::make_unique<PruneLargerFactory>();
    }else{
        prune = std::make_unique<PruneStrictLargerFactory>();
    }
    pbb->set_pruning_factory(prune);

    //SET BRANCHING
    std::unique_ptr<BranchingFactoryInterface> branch;
    branch = std::make_unique<PFSPBranchingFactory>();
    pbb->set_branching_factory(branch);

    //BUILD INITIAL SOLUTION
    pbb->set_initial_solution();


    //////////////////////////////////////
    // RUN
    //////////////////////////////////////
    enum algo{
        ivm_seqbb,
        ivm_multicore,
    };

    int choice=ivm_multicore;
	if(arguments::nbivms_mc==1)
		choice=ivm_seqbb;

    pbb->ttm->on(pbb->ttm->wall);

    switch(choice){
        case ivm_seqbb:
        {
            std::cout<<" === Run single-threaded IVM-BB"<<std::endl;

            sequentialbb<int> sbb(pbb,pbb->size);
            sbb.setRoot(pbb->root_sltn->perm);
            sbb.initFullInterval();
            sbb.run();

            break;
        }
		case ivm_multicore:
		{
			std::cout<<" === Run multi-core IVM-based BB ..."<<std::endl;

            int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
            matrix_controller mc(pbb,nthreads);

            switch (arguments::mc_ws_select) {
                case 'r':
                {
                    mc.set_victim_select(std::make_unique<RingVictimSelector>(nthreads));
                    break;
                }
                case 'a':
                {
                    mc.set_victim_select(std::make_unique<RandomVictimSelector>(nthreads));
                    break;
                }
                case 'o':
                {
                    mc.set_victim_select(std::make_unique<HonestVictimSelector>(nthreads));
                    break;
                }

            }

		    mc.initFullInterval();
			mc.next();

			break;
		}
    }
	pbb->printStats();

    pbb->ttm->off(pbb->ttm->wall);
    pbb->ttm->printElapsed(pbb->ttm->wall,"Walltime");

    return EXIT_SUCCESS;
} // main
