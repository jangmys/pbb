#include "log.h"

#include "arguments.h"
#include "pbab.h"
#include "ttime.h"

#include <sys/sysinfo.h>

//INCLUDE INSTANCES
#include "libbounds.h"

#include <mpi.h>

#include "worker_mc.h"
#ifdef USE_GPU
#include "worker_gpu.h"
#endif


#include "master.h"

int
main(int argc, char ** argv)
{
// -----------------------MPI-----------------------
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided != MPI_THREAD_SERIALIZED){
        std::cout<<"need at least MPI_THREAD_SERIALIZED support \n";
        exit(-1);
    }

    int nProc;
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);

    if(myrank==0)
        std::cout<<"-----MPI initialized with "<<nProc<<" procs-----\n";

// -----------------------Parse args----------------------
// .. pass ini file with -f <path-to-ini-file> option !
    arguments::readIniFile();
    arguments::parse_arguments(argc, argv);

// --------------------Set up logging--------------------
    FILELog::ReportingLevel() = logINFO;
#ifndef NDEBUG
    FILELog::ReportingLevel() = logDEBUG;
    std::cout<<"RUNNING IN DEBUG MODE\n";
#endif

    char buf[100];
    if (myrank == 0) {
        snprintf(buf, sizeof(buf), "./logs/%s_master.txt", arguments::inst_name);
        FILE* log_fd = fopen( buf, "w" );
        Output2FILE::Stream() = log_fd;

        std::cout<<"-----Coordinator logfile: "<<std::string(buf)<<"-----\n";
    }else{
        snprintf(buf, sizeof(buf), "./logs/%s_worker%d.txt", arguments::inst_name, myrank);
        FILE* log_fd = fopen( buf, "w" );
        Output2FILE::Stream() = log_fd;

        FILE_LOG(logINFO) << "Worker start logging";
    }

//---------------------Configure B&B---------------------
    pbab* pbb = new pbab(        pbb_instance::make_inst(arguments::problem, arguments::inst_name));

    //SET INSTANCE
    // pbb->set_instance(
    //     pbb_instance::make_instance(arguments::problem, arguments::inst_name)
    // );
    if(myrank==0){
        std::cout<<"\t#Problem:\t\t"<<arguments::problem<<" / Instance"<<arguments::inst_name<<"\n";
        std::cout<<"\t#ProblemSize:\t\t"<<pbb->size<<std::endl;
    }


    //MAKE INITIAL SOLUTION (rank 0 --> could run multiple and min-reduce...)
    if(myrank==0){
        FILE_LOG(logINFO) <<"----Initialize incumbent----";
        struct timespec t1,t2;
        clock_gettime(CLOCK_MONOTONIC,&t1);

        pbb->set_initial_solution();

        pbb->best_found.save();
        FILE_LOG(logINFO) << "Initial solution:\t" <<pbb->best_found;

        clock_gettime(CLOCK_MONOTONIC,&t2);
        FILE_LOG(logINFO) <<"Time(InitialSolution):\t"<<(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(pbb->best_found.initial_perm.data(), pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&pbb->best_found.initial_cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Bcast(pbb->best_found.perm.data(), pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&pbb->best_found.cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }else{
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(pbb->best_found.initial_perm.data(), pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&pbb->best_found.initial_cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Bcast(pbb->best_found.perm.data(), pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&pbb->best_found.cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

        FILE_LOG(logINFO) << "Initial solution:\t" <<pbb->best_found;

        MPI_Barrier(MPI_COMM_WORLD);

        // pbb->best_found.initial_cost = pbb->sltn->cost;
    }

    //LOWER BOUND
    // if(arguments::problem[0]=='f'){
    //     pbb->set_bound_factory(std::make_unique<BoundFactory>(arguments::earlyStopJohnson,arguments::johnsonPairs));
    // }else if(arguments::problem[0]=='d'){
    //     pbb->set_bound_factory(std::make_unique<DummyBoundFactory>());
    // }
    //
    // //PRUNING
    // if(arguments::findAll){
    //     pbb->choose_pruning(pbab::prune_greater);
    // }else{
    //     pbb->choose_pruning(pbab::prune_greater_equal);
    // }
    //
    // //BRANCHING
    // std::cout<<"Rank "<<myrank<<" Branching:\t"<<arguments::branchingMode<<" "<<pbb->best_found.initial_cost<<std::endl;
    // pbb->set_branching_factory(std::make_unique<PFSPBranchingFactory>(
    //     arguments::branchingMode,
    //     pbb->size,
    //     pbb->best_found.initial_cost
    // ));

    enum bb_mode{
        STANDARD,
        ITERATE_INCREASING_UB
    };

    // int bbmode=STANDARD;
    int bbmode=ITERATE_INCREASING_UB;

    switch (bbmode) {
        case STANDARD:
        {
            if (myrank == 0) {
                master mstr(pbb);

                struct timespec tstart, tend;
                clock_gettime(CLOCK_MONOTONIC, &tstart);

                mstr.initWorks(arguments::initial_work);//3 = cut initial interval in nProc pieces

                //make sure all workers have initializedMPI_Bcast(pbb->best_found.perm.data(), pbb->size, MPI_INT, 0, MPI_COMM_WORLD); pbb
                MPI_Barrier(MPI_COMM_WORLD);

                mstr.run();

                clock_gettime(CLOCK_MONOTONIC, &tend);
                printf("Walltime :\t %2.8f\n", (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9f);
            }
            else
            {
                //make sure all workers have initialized pbb
                MPI_Barrier(MPI_COMM_WORLD);

                char hostname[1024];
            	hostname[1023] = '\0';
            	gethostname(hostname, 1023);
            	FILE_LOG(logINFO) << "Worker running on :\t"<<hostname<<std::flush;

                // ==========================
                //
                worker *wrkr;
                #ifdef USE_GPU
                if(arguments::worker_type=='g'){
                    wrkr = new worker_gpu(pbb,arguments::nbivms_gpu);
                }else{
                    int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
                    wrkr = new worker_mc(pbb,nthreads);
                }
                #else
                int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
                wrkr = new worker_mc(pbb,nthreads);
                #endif
                //
                // FILE_LOG(logINFO) << "Worker running with "<<nthreads<<" threads.\n";
                //
                wrkr->run();
                //
                delete wrkr;
            }
            break;
        }
        case ITERATE_INCREASING_UB:
        {
            if (myrank == 0) {
                master mstr(pbb);

                struct timespec tstart, tend;
                clock_gettime(CLOCK_MONOTONIC, &tstart);

                MPI_Barrier(MPI_COMM_WORLD);

                int continueBB=1;
                while(continueBB){
                    pbb->ttm->reset();
                    mstr.initWorks(arguments::initial_work);//3 = cut initial interval in nProc pieces
                    mstr.reset();

                    pbb->best_found.cost++;
                    pbb->best_found.initial_cost++;

                    MPI_Bcast(pbb->best_found.perm.data(), pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pbb->best_found.cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                    MPI_Barrier(MPI_COMM_WORLD);

                    mstr.run();
                    continueBB = !pbb->best_found.foundAtLeastOneSolution;
                    MPI_Bcast(&continueBB, 1, MPI_INT, 0, MPI_COMM_WORLD);
                }

                clock_gettime(CLOCK_MONOTONIC, &tend);
                printf("Walltime :\t %2.8f\n", (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9f);
            }
            else
            {
                MPI_Barrier(MPI_COMM_WORLD);

                char hostname[1024];
            	hostname[1023] = '\0';
            	gethostname(hostname, 1023);
            	FILE_LOG(logINFO) << "Worker running on :\t"<<hostname<<std::flush;

                worker *wrkr;
                #ifdef USE_GPU
                if(arguments::worker_type=='g'){
                    wrkr = new worker_gpu(pbb,arguments::nbivms_gpu);
                }else{
                    int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
                    wrkr = new worker_mc(pbb,nthreads);
                }
                #else
                int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
                wrkr = new worker_mc(pbb,nthreads);
                FILE_LOG(logINFO) << "Worker running with "<<nthreads<<" threads.\n";
                #endif

                int continueBB=1;
                while(continueBB){
                    wrkr->reset();

                    MPI_Bcast(pbb->best_found.perm.data(), pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pbb->best_found.cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                    MPI_Barrier(MPI_COMM_WORLD);
                    wrkr->run();
                    MPI_Bcast(&continueBB, 1, MPI_INT, 0, MPI_COMM_WORLD);
                }

                delete wrkr;
            }
            break;
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return EXIT_SUCCESS;
} // main
