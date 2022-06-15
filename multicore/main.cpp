#include "arguments.h"
#include "pbab.h"
#include "solution.h"
#include "log.h"
#include "ttime.h"

#include "libbounds.h"
#include "matrix_controller.h"
#include "pool_controller.h"

#include "../ivm/intervalbb.h"
#include "../ivm/intervalbb_incr.h"
#include "../ivm/intervalbb_easy.h"

#include "../ll/pool.h"
#include "../ll/poolbb.h"

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
    InstanceFactory inst_factory;
    std::unique_ptr<instance_abstract> inst = inst_factory.make_instance(arguments::problem, arguments::inst_name);
    // set_instance(arguments::problem, arguments::inst_name);

    pbab * pbb = new pbab(inst);

    //each thread should have a private copy of the b&b operators.
    //we'll just define factory methods here that will be passed to each thread through the pbab class
    //each thread will build it's own bound, branch and prune operators later.

    //SET BOUND
    std::unique_ptr<BoundFactoryInterface<int>> bound;
    if(arguments::problem[0]=='f'){
        // pbb->set_bound_factory(std::make_unique<PFSPBoundFactory<int>>());
        bound = std::make_unique<PFSPBoundFactory<int>>();
    }else if(arguments::problem[0]=='d'){
        std::cout<<"dummy\n";
    }
    pbb->set_bound_factory(std::move(bound));

    //SET PRUNING
    if(arguments::findAll){
        pbb->set_pruning_factory(std::make_unique<PruneLargerFactory>());
    }else{
        pbb->set_pruning_factory(std::make_unique<PruneStrictLargerFactory>());
    }

    //SET BRANCHING
    pbb->set_branching_factory(std::make_unique<PFSPBranchingFactory>(
        arguments::branchingMode,
        pbb->size,
        pbb->initialUB
    ));

    //BUILD INITIAL SOLUTION
    pbb->set_initial_solution();

    //////////////////////////////////
    // RUN
    //////////////////////////////////
    enum algo{
        ivm_seqbb,
        ll_sequential,
        ivm_multicore,
        ll_multicore
    };

    int choice=ivm_multicore;
	if(arguments::nbivms_mc==1)
		choice=ivm_seqbb;

    // choice=ll_sequential;
    // choice=ll_multicore;

    pbb->ttm->on(pbb->ttm->wall);

    switch(choice){
        case ivm_seqbb:
        {
            std::cout<<" === Run single-threaded IVM-BB"<<std::endl;

            std::unique_ptr<Intervalbb<int>>sbb;

            if(arguments::boundMode == 0){
                sbb = std::make_unique<IntervalbbIncr<int>>(pbb);
            }else if(arguments::boundMode == 2){
                sbb = std::make_unique<IntervalbbEasy<int>>(pbb);
            }else{
                sbb = std::make_unique<Intervalbb<int>>(pbb);
            }

            // Intervalbb<int> sbb(
            //     pbb,
            //     pbb->branching_factory->make_branching(
            //         arguments::branchingMode,
            //         pbb->size,
            //         pbb->initialUB
            //     ),
            //     pbb->pruning_factory->make_pruning()
            // );
            sbb->setRoot(pbb->root_sltn->perm,-1,pbb->size);
            sbb->initFullInterval();
            sbb->run();

            break;
        }
        case ll_sequential:
        {
            std::cout<<" === Run single-threaded POOL-BB"<<std::endl;

            Pool p(pbb->size);

            p.set_root(pbb->root_sltn->perm,-1,pbb->size);


            std::unique_ptr<Poolbb>sbb;
            sbb = std::make_unique<Poolbb>(pbb);

            sbb->set_root(*(pbb->root_sltn));
            sbb->run();

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
        case ll_multicore:
        {
            std::cout<<" === Run multi-core LL-based BB ..."<<std::endl;

            int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
            PoolController pc(pbb,nthreads);

            pc.next();
        }
    }
	pbb->printStats();

    pbb->ttm->off(pbb->ttm->wall);
    pbb->ttm->printElapsed(pbb->ttm->wall,"Walltime");

    return EXIT_SUCCESS;
} // main
