#include <sys/sysinfo.h>

#include "arguments.h"
#include "pbab.h"
#include "log.h"
#include "ttime.h"

#include "libbounds.h"
#include "libheuristic.h"
// #include "matrix_controller.h"
#include "pool_controller.h"

// #include "../ivm/intervalbb.h"

#include "make_ll_algo.h"
// #include "make_ivm_algo.h"

int
main(int argc, char ** argv)
{
    //------------------PARAMETER PARSING-----------------
    arguments::parse_arguments(argc, argv);

    //------------------SET UP LOGGING--------------------
    FILELog::ReportingLevel() = logINFO;
#ifndef NDEBUG
    FILELog::ReportingLevel() = logDEBUG;
    std::cout<<"RUNNING IN DEBUG MODE\n";
#endif
    FILE* log_fd = fopen(arguments::logfile, "w" );
    Output2FILE::Stream() = log_fd;

    //------------------B&B components-------------------
    auto inst = pbb_instance::make_inst(arguments::problem, arguments::inst_name);

    std::shared_ptr<pbab> pbb = std::make_shared<pbab>(inst);

    //------------------BUILD INITIAL SOLUTION------------------
    pbb->set_initial_solution();

    //--------------------------Summary--------------------------
    arguments::arg_summary();

    std::cout<<"#ProblemSize:\t\t"<<pbb->size<<"\n"<<std::endl;
    std::cout<<"==========================\n";
    std::cout<<"#Initial solution\n"<<pbb->best_found;
    std::cout<<"==========================\n";

    //---------------------RUN-------------------------------------
    pbb->ttm->on(pbb->ttm->wall);
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

    std::cout<<"stop\n";
	pbb->printStats();
    pbb->ttm->off(pbb->ttm->wall);
    pbb->ttm->printElapsed(pbb->ttm->wall,"Walltime");

    return EXIT_SUCCESS;
} // main
