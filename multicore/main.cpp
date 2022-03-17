#include "arguments.h"
#include "pbab.h"
#include "solution.h"
#include "log.h"
#include "ttime.h"

#include "libbounds.h"
#include "matrix_controller.h"
#include "../ivm/sequentialbb.h"

int
main(int argc, char ** argv)
{
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

    /*SET UP LOGGING*/
    FILELog::ReportingLevel() = logERROR;
#ifndef NDEBUG
    FILELog::ReportingLevel() = logDEBUG;
#endif
    FILE* log_fd = fopen(arguments::logfile, "w" );
    Output2FILE::Stream() = log_fd;

    //pruning
    std::unique_ptr<PruningFactoryInterface> prune;
    if(arguments::findAll){
        prune = std::make_unique<PruneLargerFactory>();
    }else{
        prune = std::make_unique<PruneStrictLargerFactory>();
    }

    //branching
    std::unique_ptr<BranchingFactoryInterface> branch;
    branch = std::make_unique<PFSPBranchingFactory>();
    // if(arguments::problem[0] == 'f'){
    // }

    pbab * pbb = new pbab();
    pbb->set_pruning_factory(std::move(prune));
    pbb->set_branching_factory(std::move(branch));



    pbb->set_initial_solution();

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
