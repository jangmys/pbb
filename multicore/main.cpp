#include "arguments.h"
#include "pbab.h"
#include "solution.h"
#include "log.h"

#include "libbounds.h"
#include "matrix_controller.h"
#include "../ivm/sequentialbb.h"

int
main(int argc, char ** argv)
{
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

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

    pbab * pbb = new pbab();

    /*SET UP LOGGING*/
    FILE* log_fd = fopen(arguments::logfile, "w" );
    Output2FILE::Stream() = log_fd;

    if(arguments::init_mode==0){
        FILE_LOG(logINFO) << "Initializing at optimum " << arguments::initial_ub;
        FILE_LOG(logINFO) << "Guiding solution " << *(pbb->sltn);
        pbb->sltn->cost = arguments::initial_ub;
    }else{
        FILE_LOG(logINFO) << "Start search with heuristic solution\n" << *(pbb->sltn);
    }
    *(pbb->root_sltn) = *(pbb->sltn);

	std::cout<<"Initial solution:\n"<<*(pbb->sltn)<<"\n";


    enum algo{
        ivm_seqbb,
        ivm_multicore,
    };

    int choice=ivm_multicore;
	if(arguments::nbivms_mc==1)
		choice=ivm_seqbb;

	struct timespec tstart, tend;
	clock_gettime(CLOCK_MONOTONIC, &tstart);

    switch(choice){
        case ivm_seqbb:
        {
            std::cout<<"=== Single-threaded IVM-BB\n";
            sequentialbb *sbb=new sequentialbb(pbb,pbb->size);

            sbb->setRoot(pbb->root_sltn->perm);
            sbb->initFullInterval();

			// bool foundNew=false;

            while(sbb->next());

            delete sbb;
            break;
        }
		case ivm_multicore:
		{
			std::cout<<"=== Multi-core IVM ...\n";

			matrix_controller *mc = new matrix_controller(pbb);
			mc->initFullInterval();
			mc->next();

			delete mc;
			break;
		}
    }
	clock_gettime(CLOCK_MONOTONIC, &tend);

	pbb->printStats();

    // clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("\nWalltime :\t %2.8f\n", (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9f);

    return EXIT_SUCCESS;
} // main
