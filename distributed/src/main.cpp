#include "log.h"

#include "arguments.h"
#include "pbab.h"
#include "ttime.h"
#include "solution.h"

//INCLUDE INSTANCES
#include "libbounds.h"

#include <mpi.h>

#ifdef USE_GPU
#include "worker_gpu.h"
#else
#include "worker_mc.h"
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
    arguments::parse_arguments(argc, argv);

// --------------------Set up logging--------------------
    FILELog::ReportingLevel() = logINFO;
#ifndef NDEBUG
    FILELog::ReportingLevel() = logDEBUG;
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
    pbab* pbb = new pbab();

    //SET INSTANCE
    pbb->set_instance(
        pbb_instance::make_instance(arguments::problem, arguments::inst_name)
    );
    if(myrank==0){
        std::cout<<"\t#Problem:\t\t"<<arguments::problem<<" / Instance"<<arguments::inst_name<<"\n";
        std::cout<<"\t#ProblemSize:\t\t"<<pbb->size<<std::endl;
    }

    //LOWER BOUND
    if(arguments::problem[0]=='f'){
        pbb->set_bound_factory(std::make_unique<BoundFactory>());
    }else if(arguments::problem[0]=='d'){
        pbb->set_bound_factory(std::make_unique<DummyBoundFactory>());
    }

    //PRUNING
    if(arguments::findAll){
        pbb->choose_pruning(pbab::prune_greater);
    }else{
        pbb->choose_pruning(pbab::prune_greater_equal);
    }

    //BRANCHING
    pbb->set_branching_factory(std::make_unique<PFSPBranchingFactory>(
        arguments::branchingMode,
        pbb->size,
        pbb->initialUB
    ));

    //MAKE INITIAL SOLUTION (rank 0 --> could run multiple and min-reduce...)
    if(myrank==0){
        FILE_LOG(logINFO) <<"----Initialize incumbent----";
        struct timespec t1,t2;
        clock_gettime(CLOCK_MONOTONIC,&t1);

        pbb->set_initial_solution();

        pbb->sltn->save();
        FILE_LOG(logINFO) << "Initial solution:\t" <<*(pbb->sltn);

        clock_gettime(CLOCK_MONOTONIC,&t2);
        FILE_LOG(logINFO) <<"Time(InitialSolution):\t"<<(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9;
    }

    enum bb_mode{
        STANDARD,
        ITERATE_INCREASING_UB
    };

    int bbmode=STANDARD;

    switch (bbmode) {
        case STANDARD:
        {
            if (myrank == 0) {
                master* mstr = new master(pbb);

                struct timespec tstart, tend;
                clock_gettime(CLOCK_MONOTONIC, &tstart);

                mstr->initWorks(arguments::initial_work);//3 = cut initial interval in nProc pieces

                //make sure all workers have initialized pbb
                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Bcast(pbb->root_sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->root_sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                MPI_Bcast(pbb->sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                mstr->run();

                clock_gettime(CLOCK_MONOTONIC, &tend);
                printf("\n = Walltime :\t %2.8f\n", (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9f);
            }
            else
            {
                //make sure all workers have initialized pbb
                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Bcast(pbb->root_sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->root_sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                MPI_Bcast(pbb->sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                char hostname[1024];
            	hostname[1023] = '\0';
            	gethostname(hostname, 1023);
            	std::cout<<"Worker running on :\t"<<hostname<<std::flush;

                // ==========================
                #ifdef USE_GPU
                worker *wrkr = new worker_gpu(pbb);
                #else
                int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
                worker *wrkr = new worker_mc(pbb,nthreads);
                FILE_LOG(logINFO) << "Worker running with "<<nthreads<<" threads.\n";
                #endif

                wrkr->run();
            }
            break;
        }
        case ITERATE_INCREASING_UB:
        {
            if (myrank == 0){
                master* mstr = new master(pbb);

                pbb->buildInitialUB();
                printf("Initial Solution:\n");
                pbb->sltn->print();

                struct timespec tstart, tend;
                clock_gettime(CLOCK_MONOTONIC, &tstart);

                int continueBB=1;
                while(continueBB){
                    pbb->ttm->reset();

                    mstr->initWorks(3);
                    mstr->reset();

                    pbb->sltn->cost++;
                    MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Bcast(pbb->sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pbb->sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    mstr->run();

                    MPI_Barrier(MPI_COMM_WORLD);
                    continueBB=!pbb->foundAtLeastOneSolution;
                    MPI_Bcast(&continueBB, 1, MPI_INT, 0, MPI_COMM_WORLD);
                }

                clock_gettime(CLOCK_MONOTONIC, &tend);
                printf("\nWalltime :\t %2.8f\n", (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9f);
            }
            else
            {
                // ==========================
                #ifdef USE_GPU
                worker *wrkr = new worker_gpu(pbb);
                #else
                int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
                worker *wrkr = new worker_mc(pbb,nthreads);
                #endif

                int continueBB=1;
                while(continueBB){
                    wrkr->reset();
                    MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Bcast(pbb->sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Bcast(&pbb->sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    wrkr->run();

                    MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Bcast(&continueBB, 1, MPI_INT, 0, MPI_COMM_WORLD);
                }
            }
            break;
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return EXIT_SUCCESS;
} // main
