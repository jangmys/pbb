#include <unistd.h>
#include <mpi.h>

#include "works.h"
#include "master.h"
#include "work.h"
#include "communicator.h"

#include "macros.h"
#include "solution.h"
#include "pbab.h"
#include "ttime.h"

#include "log.h"

#include "gmp.h"
#include "gmpxx.h"

#define MAX_INTERVALS 16384 //max buffersize

master::master(pbab* _pbb) : pbb(_pbb)
{
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);

    comm = new communicator(MAX_INTERVALS,pbb->size);
    wrks = new works();

    wrk = std::make_shared<work>();

    reset();
}

master::~master()
{
    delete comm;
    delete wrks;
}

void
master::reset()
{
    end=false;
    first=true;
	stopSharing=false;

    pbb->foundAtLeastOneSolution.store(false);
    pbb->stats.totDecomposed = 0;
    pbb->stats.leaves = 0;

    pbb->ttm->reset();
}

void
master::initWorks(int initMode)
{
    wrks->clear();

    switch(initMode){
        case 1:
            wrks->init_complete(pbb);
            break;
        case 2:
            wrks->init_infile(pbb);
            break;
        case 3:
            wrks->init_complete_split(pbb,(nProc-1));
            break;
		case 4:
            wrks->init_complete_split_lop(pbb,100*(nProc-1));
            break;
        default:
            printf("master fails work init\n");
            break;
    }

    FILE_LOG(logINFO) << "Start with "<< *(pbb->root_sltn);

    end = false;
}

//InOut : w
//work unit received from worker
//Out : terminate (true if termination condition met)
//Return : True if worker needs update

static bool debug = false;
bool master::processRequest(std::shared_ptr<work> w){ //, bool &terminate) {
    bool updateWorker=false;    //return true if worker needs update

    //DEBUG
    if (debug) {
        std::cout << "work in : \n (" << (w->Uinterval).size() << "/" << w->max_intervals << ") | ID=" << w->id << std::endl;
        w->displayUinterval();
    }

    //find copy of work in works by its ID
    std::shared_ptr<work> tmp = wrks->id_find(w->id);

	bool steal=false;
    if(tmp == nullptr){
        //work with requested ID doesn't exist
        steal=true;
    }else{
        w->nb_updates++;
        tmp->nb_updates++;

        //intersection: const w, tmp can be changed and become empty
        if (w->isEmpty()){//trivial:intersect is empty
            tmp->Uinterval.clear();
        }else if(!tmp->end_updated){//non-trivial
            //copy wasn't modified : just replace
            tmp->Uinterval=w->Uinterval;
            wrks->sizes_update(tmp);
        }else{
			FILE_LOG(logDEBUG1) << "Full intersection";
            updateWorker=tmp->intersection(w);
            wrks->sizes_update(tmp);
        }

        //if result of intersection is empty work
        if(tmp->isEmpty()){
            wrks->sizes_delete(tmp);
            wrks->id_delete(tmp);
            steal=true;
        }
    }

    //if result of intersection is empty...
    if (steal) {
        if(wrks->isEmpty()){
            FILE_LOG(logDEBUG1) << "SHUTDOWN";
            end = true;
            return false;//true;
        }

        tmp = wrks->acquireNewWork(w->max_intervals,stopSharing);//terminate);
        FILE_LOG(logDEBUG4) << "#unassigned " << wrks->unassigned.size();

        if(tmp==nullptr){
            //steal may fail ...
            return false;
        }else{
            //steal succeeded ...
            updateWorker = true;
        }
    }

    w->Uinterval = tmp->Uinterval;
    w->id = tmp->id;
    tmp->end_updated=false;

    FILE_LOG(logINFO) << "ActiveSize: "<<wrks->size<<"\t Remain#: "<<wrks->unassigned.size()
	<<"\t Active#: "<<wrks->ids.size();

    //DEBUG
    if (debug) {
        printf("%s", updateWorker ? "changed...\n" : "unchanged...\n");
                std::cout << "work out : " << (w->Uinterval).size() << " items | ID=" << w->id << std::endl; w->displayUinterval();
    }

    return updateWorker;
}

//static bool first=true;
void master::shutdown() {
    if(first){
        std::cout<<" = master:\t shutting down\n";

        first=false;
        end = true;

        std::cout << " = master:\t #processed work-in messages: "<<work_in<<"\n";
        std::cout << " = master:\t #processed work-out messages: "<<work_out<<std::endl;

        std::cout<<"MASTER %\t\t:\t"<<pbb->ttm->masterLoadPerc()<<std::endl;
        pbb->ttm->printElapsed(pbb->ttm->processRequest,"ProcessREQUEST\t");

        pbb->printStats();
    }else{
        std::cout<<" = master:\t shutting down\n";
        return;
    }
}

//main thread of proc 0...
void
master::run()
{
    pbb->ttm->on(pbb->ttm->wall);

    MPI_Status status;

    int count_out=0;
    int aaa=999;
    int iter=0;

    work_in=0; work_out=0;
	solution* sol_buf=new solution(pbb->size);

    do{
        MPI_Probe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        iter++;
        pbb->ttm->on(pbb->ttm->masterWalltime);

        switch (status.MPI_TAG) {
            case WORK:
            {
                work_in++;
                wrk->clear();
                comm->recv_work(wrk, status.MPI_SOURCE, WORK, &status);

                FILE_LOG(logDEBUG1) << "Receive node count: " << wrk->exploredNodes;
                pbb->stats.totDecomposed += wrk->exploredNodes;
                pbb->stats.leaves += wrk->nbLeaves;

                pbb->ttm->on(pbb->ttm->processRequest);
                bool modified=processRequest(wrk);//,shutdownWorker);
                pbb->ttm->off(pbb->ttm->processRequest);

                //END
                if(end){
                    FILE_LOG(logDEBUG1) << "send termination signal to " << status.MPI_SOURCE;
                    MPI_Send(&aaa,1,MPI_INT,status.MPI_SOURCE,END,MPI_COMM_WORLD);
                }else if(modified){
                    //WORK
                    FILE_LOG(logDEBUG1) << "send work to " << status.MPI_SOURCE;
                    // printf("send WORK to %d\n",status.MPI_SOURCE);//status.MPI_SOURCE);
                    comm->send_work(wrk,status.MPI_SOURCE, WORK);
                    work_out++;
                // }
				// else if(stopSharing){
                //     FILE_LOG(logDEBUG1) << "send SLEEP! to " << status.MPI_SOURCE;
                //     MPI_Send(&aaa,1,MPI_INT,status.MPI_SOURCE,END,MPI_COMM_WORLD);
                //     MPI_Send(&aaa,1,MPI_INT,status.MPI_SOURCE,SLEEP,MPI_COMM_WORLD);
                }else{
                    //BEST
                    //request processed...
                    pbb->sltn->getBestSolution(sol_buf->perm,sol_buf->cost);
                    comm->send_sol(sol_buf, status.MPI_SOURCE, NIL);
                    //MPI_Send(&pbb->sltn->bestcost,1,MPI_INT,status.MPI_SOURCE,NIL,MPI_COMM_WORLD);
                }
                // FILE_LOG(logINFO) << "State\t" << 3;
                break;
            }
            case BEST:
            {
                //receive candidate solution + update Best
                solution* candidate=new solution(pbb->size);
                comm->recv_sol(candidate, status.MPI_SOURCE, BEST, &status);

                if(pbb->sltn->update(candidate->perm,candidate->cost))
                {
                    pbb->foundAtLeastOneSolution.store(true);
                    // printf("\t\tmaster_sol: %d\n",pbb->sltn->cost);
					pbb->sltn->save();
                }
                if(end){
                    FILE_LOG(logDEBUG1) << "send termination signal to " << status.MPI_SOURCE;
                    MPI_Send(&aaa,1,MPI_INT,status.MPI_SOURCE,END,MPI_COMM_WORLD);
                }else{
                    //if updatedBest
                    MPI_Send(&pbb->sltn->cost,1,MPI_INT,status.MPI_SOURCE,BEST,MPI_COMM_WORLD);
                }
                break;
            }
            case END:
            {
                MPI_Recv(&aaa, 1, MPI_INT, status.MPI_SOURCE, END, MPI_COMM_WORLD, &status);
				count_out++;
				FILE_LOG(logDEBUG1) << "...END " << status.MPI_SOURCE << " " << count_out;
                break;
            }
        }
        // printf("count %d\n",count_out);

        pbb->ttm->off(pbb->ttm->masterWalltime);

		wrks->save();

    }while(count_out!=nProc-1);//(!M->end);

    pbb->ttm->off(pbb->ttm->wall);

	sleep(1);

	FILE_LOG(logINFO) << "master iterations: "<<iter<< " master terminates...";
    FILE_LOG(logINFO) << "processed work in: "<<work_in<<"\t work out:"<<work_out;fflush(stdout);

	delete sol_buf;

    shutdown();
}
