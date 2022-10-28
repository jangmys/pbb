#include "arguments.h"
#include "pbab.h"
#include "solution.h"
#include "log.h"
#include "ttime.h"

#include "libbounds.h"
#include "libheuristic.h"
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
    //------------------PARAMETER PARSING-----------------
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

    //------------------SET UP LOGGING--------------------
    FILELog::ReportingLevel() = logERROR;
#ifndef NDEBUG
    FILELog::ReportingLevel() = logDEBUG;
#endif
    FILE* log_fd = fopen(arguments::logfile, "w" );
    Output2FILE::Stream() = log_fd;


    //------------------B&B components-------------------
    std::shared_ptr<pbab> pbb = std::make_shared<pbab>();

    //------------------SET INSTANCE----------------------
    pbb->set_instance(
        pbb_instance::make_instance(arguments::problem, arguments::inst_name)
    );

    //------------------SET BOUND-------------------------
    //each thread should have a private copy of the b&b operators.
    //we'll just define factory methods here that will be passed to each thread through the pbab class
    //each thread will build it's own bound, branch and prune operators later.
    if(arguments::problem[0]=='f'){
        pbb->set_bound_factory(std::make_unique<BoundFactory>(arguments::earlyStopJohnson,arguments::johnsonPairs));
    }else if(arguments::problem[0]=='d'){
        pbb->set_bound_factory(std::make_unique<DummyBoundFactory>());
    }

    //------------------SET PRUNING------------------
    if(arguments::findAll){
        pbb->choose_pruning(pbab::prune_greater);
    }else{
        pbb->choose_pruning(pbab::prune_greater_equal);
    }

    //------------------BUILD INITIAL SOLUTION------------------
    pbb->set_initial_solution();

    //------------------SET BRANCHING------------------
    std::cout<<"Branching:\t"<<arguments::branchingMode<<" "<<pbb->initialUB<<std::endl;
    pbb->set_branching_factory(std::make_unique<PFSPBranchingFactory>(
        arguments::branchingMode,
        pbb->size,
        pbb->initialUB
    ));


    //------------------CHOOSE ALGORITHM-----------------------
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

    // if(arguments::problem[0]=='d'){
    //     exit(0);
    // }
    //---------------------RUN-------------------------------------
    switch(choice){
        case ivm_seqbb:
        {
            std::cout<<" === Run single-threaded IVM-BB"<<std::endl;

            std::unique_ptr<Intervalbb<int>>sbb(
                make_interval_bb(pbb.get(),arguments::boundMode)
            );

            sbb->setRoot(pbb->root_sltn->perm,-1,pbb->size);

            std::vector<int> zeroFact(pbb->size,0);
            std::vector<int> endFact(pbb->size,0);

            for (int i = 0; i < pbb->size; i++) {
                endFact[i]  = pbb->size - i - 1;
            }
            sbb->initAtInterval(zeroFact, endFact);
            sbb->run();

            break;
        }
        case ll_sequential:
        {
            std::cout<<" === Run single-threaded POOL-BB"<<std::endl;

            Pool p(pbb->size);


            std::unique_ptr<subproblem> root( new subproblem(pbb->size) );
            for(int i=0;i<pbb->size;i++){
                root->schedule[i] = pbb->root_sltn->perm[i];
            }
            p.push(std::move(root));

            Poolbb sbb(pbb.get());
            // p.set_root(pbb->root_sltn->perm,-1,pbb->size);

            // std::unique_ptr<Poolbb>sbb;
            // sbb = std::make_unique<Poolbb>(pbb);

            sbb.set_root(*(pbb->root_sltn));
            sbb.run();

            break;
        }
		case ivm_multicore:
		{
            int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
			std::cout<<" === Run multi-core IVM-based BB with "<<nthreads<<" threads"<<std::endl;

            matrix_controller mc(pbb.get(),nthreads);

            mc.set_victim_select(make_victim_selector(nthreads,arguments::mc_ws_select));

            std::vector<int>_id(nthreads,0);
            std::vector<int>_pos(nthreads*pbb->size,0);
            std::vector<int>_end(nthreads*pbb->size,0);

            for (int i = 0; i < pbb->size; i++) {
                _end[i]  = pbb->size - i - 1;
            }

            mc.initFromFac(1,_id.data(),_pos.data(),_end.data());
            mc.next();

			break;
		}
        case ll_multicore:
        {
            std::cout<<" === Run multi-core LL-based BB ..."<<std::endl;

            int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
            PoolController pc(pbb.get(),nthreads);

            pc.next();
            break;
        }
    }

	pbb->printStats();
    pbb->ttm->off(pbb->ttm->wall);
    pbb->ttm->printElapsed(pbb->ttm->wall,"Walltime");

    return EXIT_SUCCESS;
} // main
