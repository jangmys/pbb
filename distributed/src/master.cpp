#include <unistd.h>
#include <mpi.h>

#include "works.h"
#include "master.h"
#include "work.h"
#include "communicator.h"

#include "macros.h"
#include "pbab.h"
#include "ttime.h"

#include "log.h"

#include "gmp.h"
#include "gmpxx.h"

master::master(pbab* _pbb) : pbb(_pbb),comm(pbb->size),wrks(),wrk(std::make_shared<work>()),isSharing(true)
{
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    reset();
}

void
master::reset()
{
    pbb->best_found.foundAtLeastOneSolution.store(false);
    pbb->stats.totDecomposed = 0;
    pbb->stats.leaves = 0;

    pbb->ttm->reset();
}

void
master::initWorks(int initMode)
{
    wrks.clear();

    switch(initMode){
        case 1:
            wrks.init_complete(pbb->size);
            break;
        case 2:
        {
            std::ifstream stream((std::string(arguments::work_directory) + "bab" + std::string(arguments::inst_name) + ".save").c_str());
            stream.seekg(0);

            if (stream) {
                stream >> pbb->best_found;
                std::cout<<"Root: "<<pbb->best_found<<"\n";

                uint64_t nbdecomposed;
                stream >> nbdecomposed;
                pbb->stats.simpleBounds = nbdecomposed;

                stream >> pbb->best_found;
                stream >> wrks;
                stream.close();
            }else{
                std::cout<<"error (read work file)\n";
            }
            break;
        }
        case 3:
            wrks.init_complete_split(pbb->size,(nProc-1));
            break;
		case 4:
            wrks.init_complete_split_lop(pbb->size,100*(nProc-1));
            break;
        default:
            printf("master fails work init\n");
            break;
    }
}

//InOut : w
//work unit received from worker
//Out : terminate (true if termination condition met)
//Return : Reply message type
static bool debug = false;
int master::processRequest(std::shared_ptr<work> w) {
    int return_type=NIL;

    //DEBUG
    if (debug) {
        std::cout << "work in : \n (" << (w->Uinterval).size() << "/" << w->max_intervals << ") | ID=" << w->id << std::endl;
        w->displayUinterval();
    }

    bool steal=false;

    //find copy of work in works by its ID
    std::shared_ptr<work> tmp = wrks.id_find(w->id);

    FILE_LOG(logDEBUG)<<"-----------------------";
    FILE_LOG(logDEBUG)<<"treat request WID "<<w->id;
    FILE_LOG(logDEBUG)<<"-----------------------";

    if(tmp == nullptr){
        //work with requested ID doesn't exist
        steal=true;
    }else{
        w->nb_updates++;
        tmp->nb_updates++;

        //intersection: const w, tmp can be changed and become empty
        if (w->isEmpty()){//trivial: result of intersection is empty
            FILE_LOG(logDEBUG)<<"TMP IS EMPTY "<<w->id<<"\t: "<<wrks.get_size();
            tmp->Uinterval.clear();
        }else if(!tmp->end_updated){//trivial : tmp hasn't changed since last time... replace!
            //-----------sanity check : size shouldn't increase!!!-----------
            auto tmpsz = tmp->wsize(); //master size
            auto wsz = w->wsize(); //worker size

            if(tmpsz<wsz){
                std::cout<<"xxxxxxxxxxxx M "<<tmpsz<<" > W  "<<wsz<<std::endl;
                tmp->displayUinterval();
                std::cout<<"<< M ================================ W >>\n";
                w->displayUinterval();
                std::cout<<std::endl;

                FILE_LOG(logINFO)<<"*********** "<<tmpsz<<" > "<<wsz;
                FILE_LOG(logINFO)<<*tmp;
                FILE_LOG(logINFO)<<"<< M ================================ W >>";
                FILE_LOG(logINFO)<<*w;
            }
            //---------------------------------------------------------------

            //copy wasn't modified : just replace
            tmp->Uinterval=w->Uinterval;
            tmp->end_updated=false;
            wrks.sizes_update(tmp);
            FILE_LOG(logDEBUG)<<"REPLACED "<<w->id<<"\t: "<<wrks.get_size();
            return NIL;
        }else{
            // FILE_LOG(logDEBUG) <<*tmp;//<<std::endl;
            // FILE_LOG(logDEBUG) <<"<< M ===============***================= W >>";
            // FILE_LOG(logDEBUG) <<*w;//<<std::endl;

            return_type = (tmp->intersection(w))?WORK:NIL;

            if(return_type==WORK)FILE_LOG(logDEBUG)<<"Full Intersect "<<w->id<<"\t: "<<wrks.get_size();
            if(return_type==NIL)FILE_LOG(logDEBUG)<<"Full Intersect "<<w->id<<"\t: EMPTY";

            // return_type=tmp->intersection(w)?WORK:NIL;
            wrks.sizes_update(tmp);
        }

        //if result of intersection is empty work
        if(tmp->isEmpty()){
            wrks.sizes_delete(tmp);
            wrks.id_delete(tmp);
            tmp=nullptr;
            steal=true;
            return_type=NIL;

            FILE_LOG(logDEBUG)<<"Empty ==> Steal";
        }
    }

    //if result of intersection is empty...
    if (steal) {
        FILE_LOG(logDEBUG)<<"STEAL";
        if(wrks.isEmpty()){
            FILE_LOG(logINFO)<<"END : "<<wrks.get_size();
            return END;//true;
        }else if (wrks.has_unassigned()) {
            FILE_LOG(logDEBUG)<<"Take NEW";
            tmp=wrks.adopt(w->max_intervals);
            return_type=NEWWORK;
        }else if(isSharing) {
            bool too_small;
            if(tmp){
                std::cout<<"tmp not NULL\n";
                if(!tmp->isEmpty()){
                    std::cout<<"tmp not empty\n";
                    std::cout<<*tmp<<std::endl;
                }
            }
            tmp=wrks.steal(w->max_intervals, too_small);
            FILE_LOG(logDEBUG)<<"Take STEAL";
            return_type=NEWWORK;
        }else{
            FILE_LOG(logDEBUG)<<"Send SLEEP";
            return SLEEP;
        }
        // tmp = wrks->acquireNewWork(w->max_intervals,shutdown);

        if(tmp==nullptr){
            FILE_LOG(logDEBUG)<<"Stolen work is NULL";
            return NIL;
        }
    }

    w->Uinterval = tmp->Uinterval;
    w->id = tmp->id;
    tmp->end_updated=false;

    //---------------------------------output---------------------------------
    FILE_LOG(logINFO) << "ActiveSize: "<<wrks.get_size()<<"\t Remain#: "<<wrks.get_num_unassigned()
	<<"\t Active#: "<<wrks.get_num_works();
    // FILE_LOG(logINFO) << "WORKSIZE : "<<wrks->size<<std::endl;

    //DEBUG
    if (debug) {
        std::cout<<((return_type!=NIL)?"changed...\n":"unchanged...\n");
        std::cout << "work out : " << (w->Uinterval).size() << " items | ID=" << w->id << std::endl; w->displayUinterval();
    }

    return return_type;
    // return (updateWorker?WORK:NIL);
}

void master::shutdown() {
    std::cout<<" = master:\t shutting down\n";

    std::cout<<"MASTER %\t\t:\t"<<pbb->ttm->masterLoadPerc()<<std::endl;
    pbb->ttm->printElapsed(pbb->ttm->processRequest,"ProcessREQUEST\t");
    pbb->printStats();
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

	// solution* sol_buf=new solution(pbb->size);

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
                comm.recv_work(wrk, status.MPI_SOURCE, WORK, &status);

                //update global (master) node count
                //==================================
                FILE_LOG(logDEBUG) << "Receive node count: " << wrk->nb_decomposed;
                pbb->stats.totDecomposed += wrk->nb_decomposed;
                pbb->stats.leaves += wrk->nb_leaves;

                //treat received work unit
                //=========================
                pbb->ttm->on(pbb->ttm->processRequest);
                int reply_type=processRequest(wrk);
                pbb->ttm->off(pbb->ttm->processRequest);

                //END
                switch (reply_type)
                {
                    case END: // NO MORE WORK LEFT
                    {
                        printf("send termination signal to %d\n",status.MPI_SOURCE);
                        MPI_Send(&aaa,1,MPI_INT,status.MPI_SOURCE,END,MPI_COMM_WORLD);
                        break;
                    }
                    //SEND NEW / UPDATED WORK UNIT
                    case NEWWORK:
                    case WORK:
                    {
                        // FILE_LOG(logINFO) << "send work "<<*wrk<<" to " << status.MPI_SOURCE;
                        comm.send_work(wrk,status.MPI_SOURCE, reply_type);
                        work_out++;
                        break;
                    }
                    case NIL: //SEND BESTCOST (confirm reception)
                    {
                        MPI_Send(&pbb->best_found.cost,1,MPI_INT,status.MPI_SOURCE,NIL,MPI_COMM_WORLD);
                        break;
                    }
                    case SLEEP:
                    {
                        MPI_Send(&aaa,1,MPI_INT,status.MPI_SOURCE,SLEEP,MPI_COMM_WORLD);
                        break;
                    }
                    default: {
                        std::cout<<"unknown return type\n";
                        break;
                    }
                }
                break;
            }
            case BEST:
            {
                //receive candidate solution + update Best
                int tmp_cost;
                int *tmp_perm = new int[pbb->size];

                comm.recv_sol(tmp_perm, tmp_cost, status.MPI_SOURCE, BEST, &status);

                if(pbb->best_found.update(tmp_perm,tmp_cost))
                {
                    pbb->best_found.foundAtLeastOneSolution.store(true);
                    pbb->best_found.save();
                    MPI_Send(&pbb->best_found.cost,1,MPI_INT,status.MPI_SOURCE,BEST,MPI_COMM_WORLD);
                }else{
                    //if updatedBest
                    MPI_Send(&pbb->best_found.cost,1,MPI_INT,status.MPI_SOURCE,BEST,MPI_COMM_WORLD);
                }
                delete[]tmp_perm;

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

        if(pbb->ttm->period_passed(CHECKPOINT_TTIME)){
            std::cout<<"SAVE"<<std::endl;

            std::ofstream stream((std::string(arguments::work_directory) + "bab" + std::string(arguments::inst_name) + ".save").c_str());
            FILE_LOG(logINFO) << "SAVED TO DISK";

            if(stream){
                stream << pbb->best_found;
                stream << pbb->stats.simpleBounds << " ";
                stream << pbb->best_found;
                stream << wrks;
                stream.close();
            }
        }
        // wrks.save();
        iter++;
    }while(count_out!=nProc-1);//(!M->end);

    pbb->ttm->off(pbb->ttm->wall);

	usleep(100);

	FILE_LOG(logINFO) << "master iterations: "<<iter<< " master terminates...";
    FILE_LOG(logINFO) << "processed work in: "<<work_in<<"\t work out:"<<work_out;fflush(stdout);

    std::cout << " = master:\t #processed work-in messages: "<<work_in<<"\n";
    std::cout << " = master:\t #processed work-out messages: "<<work_out<<std::endl;

    shutdown();
}
