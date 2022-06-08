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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided != MPI_THREAD_SERIALIZED){
        printf("need at least MPI_THREAD_SERIALIZED support \n");
        exit(-1);
    }

    int nProc;
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);

    // initializions...
    // ==> pass ini file with -f <path-to-ini-file> option !
    arguments::parse_arguments(argc, argv);

    if (myrank == 0) {
        FILELog::ReportingLevel() = logDEBUG;

        char buf[100];
        snprintf(buf, sizeof(buf), "./logs/%s_master.txt", arguments::inst_name);
        FILE* log_fd = fopen( buf, "w" );
        Output2FILE::Stream() = log_fd;
    }else{
        FILELog::ReportingLevel() = logDEBUG;
        char buf[100];
        snprintf(buf, sizeof(buf), "./logs/%s_worker%d.txt", arguments::inst_name, myrank);
        FILE* log_fd = fopen( buf, "w" );
        Output2FILE::Stream() = log_fd;

        FILE_LOG(logINFO) << "Worker start logging";
    }

    //SET INSTANCE
    InstanceFactory inst_factory;
    std::unique_ptr<instance_abstract> inst = inst_factory.make_instance(arguments::problem, arguments::inst_name);

    pbab * pbb = new pbab(inst);

    std::unique_ptr<BoundFactoryInterface<int>> bound;
    if(arguments::problem[0]=='f'){
        // pbb->set_bound_factory(std::make_unique<PFSPBoundFactory<int>>());
        bound = std::make_unique<PFSPBoundFactory<int>>();
    }else if(arguments::problem[0]=='d'){
        std::cout<<"dummy\n";
    }
    // pbb->set_bound_factory(bound);

    //SET PRUNING
    std::unique_ptr<PruningFactoryInterface> prune;
    if(arguments::findAll){
        pbb->set_pruning_factory(std::make_unique<PruneLargerFactory>());
    }else{
        pbb->set_pruning_factory(std::make_unique<PruneStrictLargerFactory>());
    }
    pbb->set_pruning_factory(std::move(prune));

    //SET BRANCHING
    // std::unique_ptr<BranchingFactoryInterface> branch;
    // branch = std::make_unique<PFSPBranchingFactory>();
    pbb->set_branching_factory(std::make_unique<PFSPBranchingFactory>(                        arguments::branchingMode,
                            pbb->size,
                            pbb->initialUB));


    pbb->set_initial_solution();

    int bbmode=0;
    switch (bbmode) {
        case 0:
        {
            if (myrank == 0) {
                std::cout<<" === Master + "<<nProc-1<<" workers\n";
                std::cout<<" === Solving "<<arguments::problem<<" / instance "<<arguments::inst_name<<"\n";
                std::cout<<" === Problem Size:\t"<<pbb->size<<std::endl;

                master* mstr = new master(pbb);

                struct timespec tstart, tend;
                clock_gettime(CLOCK_MONOTONIC, &tstart);

                mstr->initWorks(arguments::initial_work);//3 = cut initial interval in nProc pieces

                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Bcast(pbb->root_sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->root_sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                MPI_Bcast(pbb->sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                // std::cout<<" = master:\t run ..."<<std::endl;
                mstr->run();

                clock_gettime(CLOCK_MONOTONIC, &tend);
                printf("\n = Walltime :\t %2.8f\n", (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9f);
            }
            else
            {
                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Bcast(pbb->root_sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->root_sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                MPI_Bcast(pbb->sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);
                // ==========================
                #ifdef USE_GPU
                worker *wrkr = new worker_gpu(pbb);
                #else
                int nthreads = (arguments::nbivms_mc < 1) ? get_nprocs() : arguments::nbivms_mc;
                worker *wrkr = new worker_mc(pbb,nthreads);
                #endif
                // worker *wrkr = new worker_mc(pbb);

                FILE_LOG(logDEBUG) << "Worker running";
                wrkr->run();
            }
            break;
        }
        case 1:
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
