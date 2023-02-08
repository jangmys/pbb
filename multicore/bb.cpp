#include "arguments.h"
#include "pbab.h"
#include "log.h"
#include "ttime.h"

#include "libbounds.h"
#include "libheuristic.h"
#include "matrix_controller.h"
#include "pool_controller.h"

// #include "../ivm/intervalbb.h"

#include "make_ll_algo.h"
#include "make_ivm_algo.h"

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
    std::shared_ptr<pbab> pbb = std::make_shared<pbab>(
        pbb_instance::make_inst(arguments::problem, arguments::inst_name)
    );

    //------------------BUILD INITIAL SOLUTION------------------
    pbb->set_initial_solution();

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

    pbb->ttm->on(pbb->ttm->wall);

    //---------------------RUN-------------------------------------
    switch(arguments::ds){
        case 'i': //IVM
        {
            if(arguments::nbivms_mc == 1){ //SEQUENTIAL
                auto sbb = make_ivmbb<int>(pbb.get());

                //set first line of matrix
                sbb->setRoot(pbb->best_found.initial_perm.data());

                //initial interval
                std::vector<int> zeroFact(pbb->size,0);
                std::vector<int> endFact(pbb->size,0);

                for (int i = 0; i < pbb->size; i++) {
                    endFact[i]  = pbb->size - i - 1;
                }
                sbb->initAtInterval(zeroFact, endFact);

                std::cout<<" === Run single-threaded IVM-BB"<<std::endl;
                sbb->run();
            }else{
                int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;

                matrix_controller mc(pbb.get(),nthreads);
                mc.set_victim_select(make_victim_selector(nthreads,arguments::mc_ws_select));

                //initial intervals
                std::vector<int>_id(nthreads,0);
                std::vector<int>_pos(nthreads*pbb->size,0);
                std::vector<int>_end(nthreads*pbb->size,0);

                for (int i = 0; i < pbb->size; i++) {
                    _end[i]  = pbb->size - i - 1;
                }
                mc.initFromFac(1,_id.data(),_pos.data(),_end.data());

                std::cout<<" === Run multi-core IVM-based BB with "<<nthreads<<" threads"<<std::endl;
                mc.next();
            }
            break;
        }
        case 'p': //POOL
        {
            if(arguments::nbivms_mc == 1){
                std::cout<<" === Run single-threaded POOL-BB"<<std::endl;

                auto sbb = make_poolbb(pbb.get());

                subproblem p(pbb->size,pbb->best_found.initial_perm);
                sbb->set_root(p);
                sbb->run();

                pbb->stats.totDecomposed = sbb->get_decomposed_count();
                pbb->stats.leaves = sbb->get_leaves_count();
            }else{
                std::cout<<" === Run multi-core LL-based BB ..."<<std::endl;

                int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
                PoolController pc(pbb.get(),nthreads);

                pc.set_victim_select(make_victim_selector(nthreads,arguments::mc_ws_select));

                pc.next();
            }
            break;
        }
    }

	pbb->printStats();
    pbb->ttm->off(pbb->ttm->wall);
    pbb->ttm->printElapsed(pbb->ttm->wall,"Walltime");

    return EXIT_SUCCESS;
} // main
