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
    // arguments::readIniFile();
    arguments::parse_arguments(argc, argv);

    //by default initial upper bound in INFTY
    arguments::initial_ub = INT_MAX;
    //if set, read initial UB from file
    if(arguments::init_mode == 0){
        std::cout<<"Get initial upper bound from file"<<std::endl;
        switch (arguments::inst_name[0]) {
            case 't':
            {
                arguments::initial_ub = instance_taillard::get_initial_ub_from_file(arguments::inst_name,arguments::init_mode);
                break;
            }
            case 'V':
            {
                arguments::initial_ub = instance_vrf::get_initial_ub_from_file(arguments::inst_name,arguments::init_mode);
                break;
            }
        }
    }

    pbab * pbb = new pbab();//, bound1, bound2);


    int bbmode=0;

    if (myrank == 0) {
        FILELog::ReportingLevel() = logDEBUG;

        char buf[100];
        snprintf(buf, sizeof(buf), "./logs/%s_master.txt", arguments::inst_name);
        FILE* log_fd = fopen( buf, "w" );
        // FILE* log_fd = fopen( "./logs/%s_master.txt", "w" );
        Output2FILE::Stream() = log_fd;
    }else{
        FILELog::ReportingLevel() = logDEBUG;
        char buf[100];
        snprintf(buf, sizeof(buf), "./logs/%s_worker%d.txt", arguments::inst_name, myrank);
        FILE* log_fd = fopen( buf, "w" );
        Output2FILE::Stream() = log_fd;

        FILE_LOG(logINFO) << "Worker start logging";
    }

    switch (bbmode) {
        case 0:
        {
            if (myrank == 0) {
                printf(" === Master + %d workers\n", nProc - 1);
                printf(" === solving %s / instance %s\n",arguments::problem,arguments::inst_name);
                fflush(stdout);

                master* mstr = new master(pbb);

                // pbb->buildInitialUB();

                struct timespec tstart, tend;
                clock_gettime(CLOCK_MONOTONIC, &tstart);

                // mstr->initWorks(2);//2 = read from file
                mstr->initWorks(arguments::initial_work);//3 = cut initial interval in nProc pieces

                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Bcast(pbb->root_sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->root_sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                MPI_Bcast(pbb->sltn->perm, pbb->size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&pbb->sltn->cost, 1, MPI_INT, 0, MPI_COMM_WORLD);

                mstr->run();

                clock_gettime(CLOCK_MONOTONIC, &tend);
                printf("\nWalltime :\t %2.8f\n", (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9f);
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
                worker *wrkr = new worker_mc(pbb);
                #endif
                // worker *wrkr = new worker_mc(pbb);

                FILE_LOG(logDEBUG) << "Worker running";
                printf("=== R = U = N =========\n"); fflush(stdout);
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

                //get lower bound
                // bound_abstract *bd=new bound_fsp_weak();
                // bd->set_instance(pbb->instance);
                // bd->init();
                //
                // int *perm=new int[pbb->size];
                // for(int i=0;i<pbb->size;i++)perm[i]=i;
                // int c[2];
                // bd->bornes_calculer(perm,-1,pbb->size, c, 9999999);
                // std::cout<<c[0]<<" +++++ ";
                // delete perm;
                // delete bd;

                // mstr->initWorks(2);//2 = read from file
                // mstr->initWorks(4);//3 = cut initial interval in nProc pieces

                struct timespec tstart, tend;
                clock_gettime(CLOCK_MONOTONIC, &tstart);

                // if(arguments::init_mode==0){
                //     FILE_LOG(logINFO) << "Initializing at optimum " << arguments::initial_ub;
                //     FILE_LOG(logINFO) << "Guiding solution " << *(pbb->sltn);
                //     pbb->sltn->cost = arguments::initial_ub;
                // }else{
                //     FILE_LOG(logINFO) << "Start search with heuristic solution\n" << *(pbb->sltn);
                // }

                // pbb->sltn->cost = 11250;

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
                worker *wrkr = new worker_mc(pbb);
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
