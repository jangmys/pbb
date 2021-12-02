#include "arguments.h"
#include "pbab.h"
#include "solution.h"
#include "log.h"
#include "ttime.h"

#include "libbounds.h"
#include "matrix_controller.h"
#include "../ivm/sequentialbb.h"

int
main(int argc, char ** argv)
{
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

    /*SET UP LOGGING*/
    FILELog::ReportingLevel() = logINFO;
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

    pbb->ttm->on(pbb->ttm->wall);

    switch(choice){
        case ivm_seqbb:
        {
            std::cout<<" === Run single-threaded IVM-BB"<<std::endl;

            sequentialbb<int> sbb(pbb,pbb->size);
            sbb.setRoot(pbb->root_sltn->perm);
            sbb.initFullInterval();
            sbb.run();

            break;
        }
		case ivm_multicore:
		{
			std::cout<<" === Run multi-core IVM-based BB ..."<<std::endl;

            matrix_controller mc(pbb);
			mc.initFullInterval();
			mc.next();

			break;
		}
    }
	pbb->printStats();

    pbb->ttm->off(pbb->ttm->wall);
    pbb->ttm->printElapsed(pbb->ttm->wall,"Walltime");

    return EXIT_SUCCESS;
} // main
