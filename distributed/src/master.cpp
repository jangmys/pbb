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

    isSharing = true;

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
    globalEnd=false;
    first=true;

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

    globalEnd = false;
}

//InOut : w
//work unit received from worker
//Out : terminate (true if termination condition met)
//Return : Reply message type
static bool debug = false;
int master::processRequest(std::shared_ptr<work> w) {
    bool updateWorker=false;    //return true if worker needs update

    //DEBUG
    if (debug) {
        std::cout << "work in : \n (" << (w->Uinterval).size() << "/" << w->max_intervals << ") | ID=" << w->id << std::endl;
        w->displayUinterval();
    }

    bool steal=false;

    //find copy of work in works by its ID
    std::shared_ptr<work> tmp = wrks->id_find(w->id);

    if(tmp == nullptr){
        //work with requested ID doesn't exist
        steal=true;
    }else{
        w->nb_updates++;
        tmp->nb_updates++;

        //intersection: const w, tmp can be changed and become empty
        if (w->isEmpty()){//trivial: result of intersection is empty
            tmp->Uinterval.clear();
        }else if(!tmp->end_updated){//trivial : tmp hasn't changed since last time... replace!

            //-----------sanity check : size shouldn't increase!!!-----------
            auto tmpsz = tmp->wsize(); //master size
            auto wsz = w->wsize(); //worker size

            if(tmpsz<wsz){
                std::cout<<"xxxxxxxxxxxx "<<tmpsz<<" > "<<wsz<<std::endl;
                tmp->displayUinterval();
                std::cout<<"<< M ================================ W >>\n";
                w->displayUinterval();
                std::cout<<std::endl;

                FILE_LOG(logINFO)<<"*********** "<<tmpsz<<" > "<<wsz;
                FILE_LOG(logINFO)<<*tmp;
                FILE_LOG(logINFO)<<"<< M ================================ W >>";
                FILE_LOG(logINFO)<<*w;
            }

            //copy wasn't modified : just replace
            tmp->Uinterval=w->Uinterval;
            tmp->end_updated=false;
            wrks->sizes_update(tmp);
            FILE_LOG(logDEBUG4)<<"REPLACED "<<w->id<<"\t: "<<wrks->size;
            return NIL;
        }else{
            FILE_LOG(logDEBUG4)<<"Full Intersect "<<w->id<<"\t: "<<wrks->size;

            FILE_LOG(logDEBUG4) <<*tmp;//<<std::endl;
            FILE_LOG(logDEBUG4) <<"<< M ===============***================= W >>";
            FILE_LOG(logDEBUG4) <<*w;//<<std::endl;

			// FILE_LOG(logDEBUG1) << "Full intersection";
            updateWorker=tmp->intersection(w);
            wrks->sizes_update(tmp);
        }

        //if result of intersection is empty work
        if(tmp->isEmpty()){
            wrks->sizes_delete(tmp);
            wrks->id_delete(tmp);
            tmp=nullptr;
            steal=true;
        }
    }

    //if result of intersection is empty...
    if (steal) {
        if(wrks->isEmpty()){
            FILE_LOG(logINFO)<<"END : "<<wrks->size;
            return END;//true;
        }else if (!wrks->unassigned.empty())   {
            tmp=wrks->_adopt(w->max_intervals);
            updateWorker=true;
        }else if(isSharing) {
            bool too_small;
            if(tmp){
                std::cout<<"tmp not NULL\n";
                if(!tmp->isEmpty()){
                    std::cout<<"tmp not empty\n";
                    std::cout<<*tmp<<std::endl;
                }
            }
            tmp=wrks->steal(w->max_intervals, too_small);
            updateWorker=true;
        }else{
            return SLEEP;
        }
        // tmp = wrks->acquireNewWork(w->max_intervals,shutdown);

        FILE_LOG(logDEBUG4) << "#unassigned " << wrks->unassigned.size();

        if(tmp==nullptr){
            return NIL;
        }
    }

    w->Uinterval = tmp->Uinterval;
    w->id = tmp->id;
    tmp->end_updated=false;

    //---------------------------------output---------------------------------
    FILE_LOG(logINFO) << "ActiveSize: "<<wrks->size<<"\t Remain#: "<<wrks->unassigned.size()
	<<"\t Active#: "<<wrks->ids.size();
    std::cout<<"WORKSIZE : "<<wrks->size<<std::endl;

    //DEBUG
    if (debug) {
        std::cout<<(updateWorker?"changed...\n":"unchanged...\n");
        std::cout << "work out : " << (w->Uinterval).size() << " items | ID=" << w->id << std::endl; w->displayUinterval();
    }

    return (updateWorker?WORK:NIL);
}

//static bool first=true;
void master::shutdown() {
    if(first){
        std::cout<<" = master:\t shutting down\n";

        first=false;
        // globalEnd = true;

        std::cout<<"MASTER %\t\t:\t"<<pbb->ttm->masterLoadPerc()<<std::endl;
        pbb->ttm->printElapsed(pbb->ttm->processRequest,"ProcessREQUEST\t");
        pbb->printStats();
    }else{
        std::cout<<" = master:\t shutting down\n";
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

    int work_in=0, work_out=0;

	solution* sol_buf=new solution(pbb->size);

    do{
        //waiting for any message
        MPI_Probe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

        //master active now
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
                // bool shutdown_worker=false;
                int reply_type=processRequest(wrk);
                pbb->ttm->off(pbb->ttm->processRequest);

                //END
                switch (reply_type) {
                    case END: // NO MORE WORK LEFT
                    {
                        printf("send termination signal to %d\n",status.MPI_SOURCE);
                        MPI_Send(&aaa,1,MPI_INT,status.MPI_SOURCE,END,MPI_COMM_WORLD);
                        break;
                    }
                    case WORK: //SEND UPDATED WORK UNIT
                    {
                        // FILE_LOG(logINFO) << "send work "<<*wrk<<" to " << status.MPI_SOURCE;
                        comm->send_work(wrk,status.MPI_SOURCE, WORK);
                        work_out++;
                        break;
                    }
                    case NIL: //SEND BESTCOST (confirm reception)
                    {
                        //request processed...
                        int tmp;
                        pbb->sltn->getBestSolution(sol_buf->perm,tmp);
                        sol_buf->cost.store(tmp);

                        MPI_Send(&pbb->sltn->cost,1,MPI_INT,status.MPI_SOURCE,NIL,MPI_COMM_WORLD);
                        break;
                    }
                    case SLEEP:
                    {
                        MPI_Send(&aaa,1,MPI_INT,status.MPI_SOURCE,SLEEP,MPI_COMM_WORLD);
                        break;
                    }
                    default:
                        std::cout<<"unknown return type\n";
                }
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
                    pbb->sltn->save();
                    // printf("\t\tmaster_sol: %d\n",pbb->sltn->cost);
                // }
                // if(globalEnd){
                //     FILE_LOG(logDEBUG1) << "send termination signal to " << status.MPI_SOURCE;
                //     MPI_Send(&aaa,1,MPI_INT,status.MPI_SOURCE,END,MPI_COMM_WORLD);
                    MPI_Send(&pbb->sltn->cost,1,MPI_INT,status.MPI_SOURCE,BEST,MPI_COMM_WORLD);
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
                break;
            }
        }
        pbb->ttm->off(pbb->ttm->masterWalltime);

		wrks->save();
        iter++;
    }while(count_out!=nProc-1);//(!M->end);

    pbb->ttm->off(pbb->ttm->wall);

    globalEnd=true;

	usleep(100);

	FILE_LOG(logINFO) << "master iterations: "<<iter<< " master terminates...";
    FILE_LOG(logINFO) << "processed work in: "<<work_in<<"\t work out:"<<work_out;fflush(stdout);

	delete sol_buf;

    std::cout << " = master:\t #processed work-in messages: "<<work_in<<"\n";
    std::cout << " = master:\t #processed work-out messages: "<<work_out<<std::endl;

    shutdown();
}
