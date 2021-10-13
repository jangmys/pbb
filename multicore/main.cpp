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

    /*SET UP LOGGING*/
    FILE* log_fd = fopen(arguments::logfile, "w" );
    Output2FILE::Stream() = log_fd;

    pbab * pbb = new pbab();
    pbb->set_initial_solution();

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
            std::cout<<" === Run single-threaded IVM-BB"<<std::endl;
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
			std::cout<<" === Run multi-core IVM-based BB ..."<<std::endl;

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
