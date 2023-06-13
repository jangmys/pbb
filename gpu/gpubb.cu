#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <memory>

#include "pbab.h"
#include "arguments.h"
//nbivms_gpu, problem, boundMode, branchingMode, findAll, printSolutions,ws_strategy

#include "set_operators.h"
#include "subproblem.h"
#include "ttime.h"
#include "log.h"

#include "libbounds.h"
#include "gpubb.h"

// all CUDA from this file
#include "./bb_kernels.cu"



int gpu_worksteal::steal_in_device(int* line, int* pos, int* end, int* dir, int* mat, int* state, int iter, unsigned int nb_exploring)
{
    adapt_workstealing(nb_exploring, 2, nbIVM / 8);

    computeLength <<< (nbIVM / PERBLOCK), (32 * PERBLOCK)>>>(pos, end, state, length_d, sumLength_d);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    //    search_cut = 0.1;
    computeMeanLength << < (nbIVM + 127) / 128, 128>>>
    (sumLength_d, meanLength_d, search_cut, nbIVM); // (int)(ctrl_h[2]+1));
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    int dimb = iter % topoDimensions;
    int from, to, dim, q;

    for (int s = dimb; s < dimb + topoDimensions; s++) {
        dim  = s % topoDimensions;
        q    = (1 << topoB[dim]);
        from = iter & (q - 1);
        to   = from + q;

        for (int off = from; off < to; off++)
            prepareShare << < (nbIVM + 127) / 128, 128>>>
            (state, victim_flag_d, victim_d, length_d, meanLength_d, off & (q - 1), topoB[dim], topoA[dim]);
    }
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    share_on_gpu2 <<< nbIVM / PERBLOCK, 32 * PERBLOCK>>>
    (mat, pos, end, dir, line, 1, 2, state, victim_flag_d, victim_d);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    unsigned int tmp = 0;
    gpuErrchk(cudaMemcpyFromSymbol(&tmp, gpuBalancedIntern, sizeof(unsigned int)));

    return tmp;
};



gpubb::gpubb(pbab * _pbb) : pbb(_pbb),size(pbb->size),nbIVM(arguments::nbivms_gpu)
{
    // pbb  = _pbb;
    // size = pbb->size;
    // nbIVM    = arguments::nbivms_gpu;
    // ringsize = nbIVM;



    // setHypercubeConfig(nbIVM); //work stealing

    ws = std::make_unique<gpu_worksteal>(size,nbIVM);

    if (arguments::problem[0] == 'f') {
        bound = make_bound_ptr<int>(pbb,arguments::primary_bound);
    }else{
        bound = std::make_unique<bound_dummy>();
    }

    initialUB = pbb->best_found.getBest();

	FILE_LOG(logINFO) << "GPU with nbIVM:\t" << nbIVM;
	FILE_LOG(logINFO) << "Initial UB:\t" << initialUB;

    pthread_mutex_init(&mutex_end,NULL);
    pthread_mutex_lock(&mutex_end);
    allEnd = false;
    pthread_mutex_unlock(&mutex_end);

    // "one time events"
    firstbound = true;
    // search_cut = 1.0;

	execmode.triggered = false;
}

gpubb::~gpubb()
{
    // free_on_device();
    // free etc
}

//
void
gpubb::initialize(int rank)
{
    //-----------mapping MPI_ranks to devices-----------
    int device,num_devices;
    gpuErrchk( cudaGetDeviceCount(&num_devices) );
    gpuErrchk( cudaSetDevice(rank % num_devices) );

    gpuErrchk( cudaGetDevice(&device) );
    std::cout<<rank<<" using device "<<device<<" of "<<num_devices<<"\n";

    gpuErrchk(cudaFree(0));

    // std::cout<<"initialized context\n";
    stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    event = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
    gpuErrchk(cudaStreamCreate(stream));
    gpuErrchk(cudaEventCreateWithFlags(event, cudaEventDisableTiming));

    allocate_on_host();
    allocate_on_device();

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
} // gpubb::initialize



void
gpubb::initFullInterval()
{
    if (firstbound) {
        int best = INT_MAX;
        pbb->best_found.getBest(best);
		FILE_LOG(logINFO) << "Init Full : Bound Root with UB:\t" << best;
        FILE_LOG(logINFO) << "Init Full : Root :\t" << pbb->best_found;
#ifdef FSP
		FILE_LOG(logINFO) << "Init Full : size:\t" << size << " " << nbMachines_h ;
#endif
        int* tmp_state = new int[nbIVM];
        memset(tmp_state,0,nbIVM*sizeof(int));
        tmp_state[0] = 1;
        gpuErrchk( cudaMemcpy(state_d, tmp_state, nbIVM*sizeof(int),cudaMemcpyHostToDevice) );

        delete[] tmp_state;

        // bound root node
        #ifdef FSP
        //1. set 1st row of M to bestsol
		gpuErrchk( cudaMemcpy(mat_d,pbb->best_found.initial_perm.data(),size*sizeof(int),cudaMemcpyHostToDevice) );

        //2. compute LB (begin) for all subpb
        decode(4);
        weakBound(4, best);

        int *d_root_tmp;
        int *d_root_dir_tmp;
        cudaGetSymbolAddress((void **)&d_root_tmp, root_d);
        cudaGetSymbolAddress((void **)&d_root_dir_tmp, root_dir_d);

        gpuErrchk( cudaMemcpy(d_root_tmp, mat_d, size*sizeof(int),cudaMemcpyDeviceToDevice) );
        gpuErrchk( cudaMemcpy(d_root_dir_tmp, dir_d, sizeof(int),cudaMemcpyDeviceToDevice) );
        #endif
        #ifdef TEST
        boundRoot << < 1, 128, sizeof(int) * size >>> (mat_d, dir_d, line_d);
        #endif
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // cudaFree(bestsol_d);

        gpuErrchk(cudaMemcpy(costsBE_h,costsBE_d,2*nbIVM*size*sizeof(int),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(dir_h,dir_d,nbIVM*size,cudaMemcpyDeviceToHost));

        firstbound = false;
    }

	dim3 blks((nbIVM * 32 + 127) / 128);
	setRoot << < blks, 128, 0, stream[0] >> > (mat_d, dir_d);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    memset(pos_h, 0, nbIVM * size * sizeof(int));
    memset(end_h, 0, nbIVM * size * sizeof(int));
    memset(state_h, 0, nbIVM * sizeof(int));
    memset(line_h, 0, nbIVM * sizeof(int));

	for(int i=0;i<size;i++){
		end_h[i]=size-i-1;
	}
	state_h[0] = -1;//initialize first

    copyH2D_update();

    // std::cout<<"ROOT DECOMPOSED =====================\n";
    // affiche(1);
    // std::cout<<"=====================\n";
}

void
gpubb::interruptExploration()
{
    pthread_mutex_lock(&mutex_end);
    allEnd = true;
    pthread_mutex_unlock(&mutex_end);
}

void
gpubb::selectAndBranch(const int NN)
{
    // int best = INT_MAX;
    // pbb->sltn->getBest(best);
    gpuErrchk(cudaMemset(counter_d, 0, 6 * sizeof(unsigned int)));
	//dense mapping : one thread = one IVM
    goToNext_dense<<< (nbIVM+127) / 128, 128, 0, stream[0] >>>(mat_d, pos_d, end_d, dir_d, line_d, state_d, nbDecomposed_d, counter_d, NN);

	//wide mapping : one warp = one IVM
    // assume:
    // 1 block = NN warps = NN IVM
    // size_t smem = NN * (2 * size * sizeof(int) + 2 * sizeof(int));
    // goToNext2<4><<< (nbIVM+NN-1) / NN, NN * 32, smem, stream[0] >>>(mat_d, pos_d, end_d, dir_d, line_d, state_d, nbDecomposed_d, counter_d);

#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
}

void
gpubb::launchBBkernel(const int NN)
{
    int best = INT_MAX;

    pbb->best_found.getBest(best);

    gpuErrchk(cudaMemset(counter_d, 0, 6 * sizeof(unsigned int)));
    gpuErrchk(cudaMemset(flagLeaf, 0, nbIVM * sizeof(int)));
    unsigned int target_h = 0;
    gpuErrchk(cudaMemcpyToSymbol(_trigger, &target_h, sizeof(unsigned int)));
    gpuErrchk(cudaMemcpyToSymbol(targetNode, &target_h, sizeof(unsigned int)));
    gpuErrchk(cudaMemset(costsBE_d, 999999, 2 * size * nbIVM * sizeof(int)));

#ifdef FSP
    size_t smem = NN * (3*nbMachines_h + 3 * size + 2) * sizeof(int);
    multistep_triggered<4><<< (nbIVM+NN-1) / NN, NN * 32, smem, stream[0] >>>
    (mat_d, pos_d, end_d, dir_d, line_d, state_d, nbDecomposed_d, counter_d, schedule_d, lim1_d, lim2_d,costsBE_d,flagLeaf, best, initialUB);
#else
    size_t smem = 0;
#endif

#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
}

bool
gpubb::allDone()
{
    // gpuErrchk(cudaMemcpy(state_h,state_d,nbIVM*sizeof(int),cudaMemcpyDeviceToHost));

    unsigned int end_h = 0;
    gpuErrchk(cudaMemcpyToSymbol(deviceEnd, &end_h, sizeof(unsigned int)));

    // blockReduce+atomic
    if (nbIVM >= 1024)
        checkEnd << < (nbIVM + 1023) / 1024, 1024, 0, stream[0] >>> (state_d);
    else if (nbIVM == 512)
        checkEnd << < (nbIVM + 511) / 512, 512, 0, stream[0] >> > (state_d);
    else{
        checkEnd << < (2*nbIVM - 1) / nbIVM, std::max(32,nbIVM), 0, stream[0] >> > (state_d);
    }
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    gpuErrchk(cudaMemcpyFromSymbol(&end_h, deviceEnd, sizeof(unsigned int)));
    return (end_h==0);
}

void
gpubb::buildMapping(int best)
{
    //prefix-sum

    //carries
    gpuErrchk(cudaMemset(auxArr, 0, 256 * sizeof(int)));

    dim3 blocksz(256);
    dim3 numbblocks((nbIVM + blocksz.x - 1) / blocksz.x);
    size_t smem = 2 * blocksz.x * sizeof(int);
    reduce<<< numbblocks, blocksz, smem, stream[0] >> > (todo_d, tmp_arr_d, auxArr, blocksz.x);
	reduce2<<< numbblocks, blocksz, 0, stream[0] >> > (tmp_arr_d, auxArr);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

	switch(arguments::boundMode){
		case 1:
		{
            //all
			prepareBound2<<<(nbIVM+PERBLOCK-1) / PERBLOCK, 32 * PERBLOCK, 0, stream[0]>>>(lim1_d, lim2_d, todo_d, ivmId_d, toSwap_d, tmp_arr_d,state_d);
			break;
		}
		case 2:
		{
            //conditionally on cost
	    	prepareBound<<<(nbIVM+127) / 128, 128, 0, stream[0]>>>(schedule_d, costsBE_d,dir_d,line_d,lim1_d, lim2_d, todo_d, ivmId_d, toSwap_d, tmp_arr_d, state_d,best);

            // int *swap_h = new int[size * nbIVM];
            // int *ivm_h = new int[size * nbIVM];
            //
            // cudaMemcpy(swap_h, toSwap_d, size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(ivm_h, ivmId_d, size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
            //
            // int ttodo_h;
            // gpuErrchk(cudaMemcpyFromSymbol(&ttodo_h, todo, sizeof(unsigned int)));
            //
            // printf("*** swp\t");
            // for (int j = 0; j < ttodo_h; j++) printf("%2d\t", swap_h[j]);
            // printf("\n");
            // printf("*** id\t");
            // for (int j = 0; j < ttodo_h; j++) printf("%2d\t", ivm_h[j]);
            // printf("\n");
            //
            // delete[]swap_h;
            // delete[]ivm_h;


			break;
		}
	}
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
}

//called from worker::doWork or main (single process mode)
bool
gpubb::next()
{
    bool end = false;
    int iter = 0;
    int best = INT_MAX;

    pbb->best_found.getBest(best);

    int nbsteals = 0;
	localFoundNew=false;

    // std::cout<<"got best "<<best<<std::endl;

    while (true) {
        if (counter_h[exploringState] < 75*nbIVM/100){
            nbsteals += ws->steal_in_device(line_d, pos_d, end_d, dir_d, mat_d, state_d, iter, counter_h[exploringState]);
        }

        if(execmode.triggered){
            //perform whole BB in single kernel..break if a threshold of empty explorers is reached.
            end = triggeredNext(best,iter);
        }else{
            //one BB step
            end = next(best,iter);
        }
        iter++;

        //conditions to trigger communication with master
        if(!arguments::singleNode){
            if(execmode.triggered){
                break;
            }
            if(((nbsteals > (nbIVM/5)) && iter>100) || localFoundNew || pbb->ttm->period_passed(WORKER_BALANCING)){
                break;
            }
        }
        //no more work !
        if(end){
            break;
        }
    }

    FILE_LOG(logDEBUG) << "GPU-explore : BREAK";


	//return true if allEnd
    return end;
}

// returns true iff no more work available
bool
gpubb::next(int& best, int iter)
{
    bool end = false;

    //modify IVM structures to make them point to next subproblem
    selectAndBranch(4);
    //at each iteration get some exploration statistics
    getExplorationStats(iter,best);

    //check termination
    if (allDone()){
        return true;
    }

    //get subproblems from IVM
    bool reachedLeaf=decode(4);

    //if not 'strong-bound-only' ...
	if(arguments::boundMode != 1){
		reachedLeaf = weakBound(4, best);
	}

	unsigned int target_h=0;
    gpuErrchk(cudaMemcpyFromSymbol(&target_h, targetNode, sizeof(unsigned int)));
    reachedLeaf = (bool) target_h;

    // affiche(1);

    boundLeaves(reachedLeaf,best);

    //evaluate one-by-one
    if(arguments::boundMode != 0){
        buildMapping(best);

        int ttodo_h;
        gpuErrchk(cudaMemcpyFromSymbol(&ttodo_h, todo, sizeof(unsigned int)));

        // std::cout<<"TODO_H "<<ttodo_h<<"\n";
        // affiche(1);


        cudaMemset(sums_d, 0, 2 * nbIVM * sizeof(int));
        cudaMemset(costsBE_d, 0, 2 * size * nbIVM * sizeof(int));

        dim3 boundblocks;

        switch(arguments::boundMode){
            case 1://no pre-bounding...
            {
                boundblocks.x = (2 * ttodo_h+127) / 128;
#ifdef FSP
                boundJohnson<<<boundblocks, 128, nbJob_h * nbMachines_h + 64 * nbJob_h, stream[0]>>>(schedule_d, lim1_d, lim2_d, line_d, costsBE_d, sums_d, state_d, toSwap_d,ivmId_d, nbLeaves_d, ctrl_d, flagLeaf, best);
#endif
#ifdef TEST
                /*
                boundStrong kernel here!!! (if strong bound only)
                */
#endif
                int smem = (4 * (2*size + 2) * sizeof(int));
                chooseBranchingSortAndPrune<<< (nbIVM+4-1) / 4, 4 * 32, smem, stream[0] >> >
                (mat_d, dir_d, pos_d, lim1_d, lim2_d, line_d, schedule_d, costsBE_d, prio_d, state_d, todo_d, best, initialUB,arguments::branchingMode);
                break;
            }
            case 2:
            {
                if(ttodo_h>0){
                    boundblocks.x = (ttodo_h+127) / 128;
#ifdef FSP
                    boundOne <<< boundblocks, 128, nbJob_h * nbMachines_h,stream[0] >>>
                    (schedule_d, lim1_d, lim2_d, dir_d, line_d, costsBE_d, toSwap_d, ivmId_d,best, bd->front_d, bd->back_d);
#endif
#ifdef TEST
                    /*
                    TO IMPLEMENT
                    boundStrong kernel here!!! (if mixed bound)
                    */
                    boundStrongFront <<< boundblocks, 128, 0, stream[0] >>>
                    (schedule_d, lim1_d, lim2_d, line_d, toSwap_d, ivmId_d, costsBE_d);
#endif

                    prune2noSort <<< (nbIVM+PERBLOCK-1) / PERBLOCK, 32 * PERBLOCK, 0, stream[0] >> > (mat_d, dir_d, line_d, costsBE_d, state_d, best);
                }
                break;
    	    }
        }
    }
    #ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #endif

    return end; // (ctrl_h[gpuEnd] == 1);
} // gpubb::next


bool gpubb::decode(const int NN)
{
    unsigned int target_h = 0;

	switch(arguments::boundMode){
        case 1: //strong bound
        {
            /*
            decode IVM and prepare mapping for lower bounding
            */
            gpuErrchk(cudaMemset(todo_d, 0, nbIVM * sizeof(int)));
            gpuErrchk(cudaMemset(flagLeaf, 0, nbIVM * sizeof(int)));
            gpuErrchk(cudaMemcpyToSymbol(targetNode, &target_h, sizeof(unsigned int)));

            size_t smem = (NN * (size + 3)) * sizeof(int);
            decodeIVM<<< (nbIVM+NN-1) / NN, NN * 32, smem, stream[0] >>>
                    (mat_d, dir_d, pos_d, lim1_d, lim2_d, line_d, schedule_d, state_d);

            flagLeaf_fillTodo<<<(nbIVM+127)/128,128>>>( flagLeaf, todo_d, lim1_d, lim2_d, line_d, state_d);

            gpuErrchk(cudaMemcpyFromSymbol(&target_h, targetNode, sizeof(unsigned int)));
     	    break;
        }
        case 0: //weak bound (evaluate children)
        case 2: //mixed (evaluate children)
        {
            /*
            just decode IVM
            */
            size_t smem = (NN * (size + 3)) * sizeof(int);
            decodeIVM<<< (nbIVM+NN-1) / NN, NN * 32, smem, stream[0] >>>
                (mat_d, dir_d, pos_d, lim1_d, lim2_d, line_d, schedule_d, state_d);

            break;
        }
    }

    return ((bool) target_h);
}

bool
gpubb::weakBound(const int NN, const int best)
{
    //reset flags
    unsigned int target_h = 0;
    gpuErrchk(cudaMemcpyToSymbol(targetNode, &target_h, sizeof(unsigned int)));
    gpuErrchk(cudaMemset(flagLeaf, 0, nbIVM * sizeof(int)));

    gpuErrchk(cudaMemset(costsBE_d, 0, 2 * size * nbIVM * sizeof(int))); //memset sets bytes!
	size_t smem;

#ifdef FSP
    smem = (NN * (size + 3 * nbMachines_h)) * sizeof(int);
    if(arguments::branchingMode>0){ //BEGIN-END
        boundWeak_BeginEnd<32><<<(nbIVM+NN-1) / NN, NN * 32, smem, stream[0] >>>(lim1_d, lim2_d, line_d, schedule_d, costsBE_d, state_d, bd->front_d, bd->back_d, best, flagLeaf);
    }else if(arguments::branchingMode==-2){ //FWD only
        boundWeak_Begin<32> << < (nbIVM+NN-1) / NN, NN * 32, smem, stream[0] >> >
        (lim1_d, lim2_d, line_d, schedule_d, costsBE_d, state_d, bd->front_d, bd->back_d, best, flagLeaf);
    }
#endif
#ifdef TEST
    //shared memory for schedulesnext
    smem = NN * size * sizeof(int);
    boundWeakFront<<<(nbIVM+NN-1) / NN, NN * 32, smem, stream[0] >>>(state_d,schedule_d,lim1_d,lim2_d,line_d,costsBE_d,flagLeaf);
#endif

#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    gpuErrchk(cudaMemset(todo_d, 0, nbIVM * sizeof(int)));
    if(arguments::branchingMode>0){
        //dynamic
        smem = (NN * (2*size + 2) * sizeof(int));
        chooseBranchingSortAndPrune<<< (nbIVM+NN-1) / NN, NN * 32, smem, stream[0] >> >
        (mat_d, dir_d, pos_d, lim1_d, lim2_d, line_d, schedule_d, costsBE_d, prio_d, state_d, todo_d, best, initialUB,arguments::branchingMode);
    }
    else if(arguments::branchingMode==-3){    }
    else if(arguments::branchingMode==-2){
        //forward
        smem = (NN * (2*size + 2) * sizeof(int));
        ForwardBranchSortAndPrune<<< (nbIVM+NN-1) / NN, NN * 32, smem, stream[0] >> >
        (mat_d, dir_d, pos_d, lim1_d, lim2_d, line_d, schedule_d, costsBE_d, prio_d, state_d,
            todo_d, best, flagLeaf);
    }
    else if(arguments::branchingMode==-1){
        //backward
    }

#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    gpuErrchk(cudaMemcpyFromSymbol(&target_h, targetNode, sizeof(unsigned int)));
    return ((bool) target_h);
} // gpubb::decodeAndBound


void gpubb::getExplorationStats(const int iter, const int best)
{
    // Don't remove this memcopy : used in work stealing
    gpuErrchk(cudaMemcpy(counter_h, counter_d, 6 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //just to get some feedback
    if ((arguments::gpuverb && iter%arguments::gpuverb==0) && (iter > 0)) {
        std::cout << "Iter: "<<iter<<"\tExpl: " << counter_h[exploringState] << "\tEmpty: " << counter_h[emptyState] << "\n";
		// FILE_LOG(logINFO) << "Expl: " << counter_h[exploringState] << "\tEmpty: " << counter_h[emptyState];
    }
}

bool
gpubb::boundLeaves(bool reached, int& best)
{
    if(!reached)return false;

    int flags[nbIVM];

    gpuErrchk( cudaMemcpy(flags, flagLeaf, nbIVM * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(schedule_h, schedule_d, nbIVM * size * sizeof(int), cudaMemcpyDeviceToHost) );

    bool newUB = false;

    for (int k = 0; k < nbIVM; ++k) {
        if (flags[k] == 1) {
            int cost=bound->evalSolution(schedule_h+k*size);
            pbb->stats.leaves++;
            // FILE_LOG(logINFO) << "Evaluated Leaf\t" << cost << " vs. Best "<<best;

            bool update;
            if(arguments::findAll)update=(cost<=best);
            else update=(cost<best);

            // for(int i=0;i<size;i++){
            //     std::cout<<schedule_h[k*size+i]<<" ";
            // }
            // std::cout<<std::endl;


            if (update) {
                best = cost;
                pbb->best_found.update(schedule_h+k*size,cost);

				localFoundNew = true;
                pbb->best_found.foundAtLeastOneSolution.store(true);

                FILE_LOG(logINFO) << "GPUBB found " << pbb->best_found;

                newUB = true;
            }
        }
    }

    return newUB;
}


// = =========================================================
//
//
//
//
//
//
//
//        ALLOCATION, COPY ROUTINES, ETC
//
//
//
//
//
//
// ========================================================
void
gpubb::allocate_on_host()
{
	FILE_LOG(logINFO) << "Allocate memory for "<<nbIVM<<" IVM of size " << size;

    int size_m = size * size * nbIVM;
    int size_v = size * nbIVM;
    int size_i = nbIVM;

    mat_h      = (int *) calloc(size_m, sizeof(int));
    pos_h      = (int *) calloc(size_v, sizeof(int));
    end_h      = (int *) calloc(size_v, sizeof(int));
    dir_h      = (int *) calloc(size_v, sizeof(int));
    line_h     = (int *) calloc(size_i, sizeof(int));
    state_h    = (int *) calloc(size_i, sizeof(int));
    lim1_h     = (int *) calloc(size_i, sizeof(int));
    lim2_h     = (int *) calloc(size_i, sizeof(int));
    schedule_h = (int *) calloc(size_v, sizeof(int));

    ctrl_h       = (unsigned int *) calloc(8, sizeof(unsigned int));
    counter_h    = (unsigned int *) calloc(6, sizeof(unsigned int));
    costsBE_h    = (int *) calloc(2 * size_v, sizeof(int));

    nbDecomposed_h = (unsigned long long int *) calloc(size_i, sizeof(unsigned long long int));
}

void
gpubb::allocate_on_device()
{
    int size_m = size * size * nbIVM;
    int size_v = size * nbIVM;
    int size_i = nbIVM;

    gpuErrchk(cudaMalloc((void **) &mat_d, size_m * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &pos_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &end_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &dir_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &line_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &state_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &lim1_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &lim2_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &schedule_d, size_v * sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &prio_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &costsBE_d, 2 * size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &sums_d, 2 * size_i * sizeof(int)));
    //
    gpuErrchk(cudaMalloc((void **)&split_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &victim_flag, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &victim_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &length_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &sumLength_d, size * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &meanLength_d, size * sizeof(int)));
    //
    gpuErrchk(cudaMalloc((void **) &flagLeaf, size_i * sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &auxArr, 256 * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &auxEnd, 256 * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &tmp_arr_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &todo_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &toSwap_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &ivmId_d, size_v * sizeof(int)));
    //
    gpuErrchk(cudaMalloc((void **) &ctrl_d, 4 * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **) &counter_d, 6 * sizeof(unsigned int)));
    //
    // gpuErrchk(cudaMalloc((void **) &nbDecomposed_d, size_i * sizeof(unsigned long long int)));
    // gpuErrchk(cudaMalloc((void **) &nbLeaves_d, size_i * sizeof(int)));
    //
    // // set to 0
    // gpuErrchk(cudaMemset(counter_d, 0, 6 * sizeof(unsigned int)));
    // gpuErrchk(cudaMemset(ivmId_d, 0, size_v * sizeof(int)));
    // gpuErrchk(cudaMemset(sums_d, 0, 2 * size_i * sizeof(int)));
    //
    // gpuErrchk(cudaMemset(ws.victim_flag_d, 0, size_i * sizeof(int)));
    // gpuErrchk(cudaMemset(ws.sumLength_d, 0, size * sizeof(int)));
    // unsigned int zero=0;
    // gpuErrchk(cudaMemcpyToSymbol(countNodes_d,&zero,sizeof(unsigned int)));
    //
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());
} // gpubb::allocate_on_device

void
gpubb::free_on_device()
{
    gpuErrchk(cudaFree(mat_d));
    gpuErrchk(cudaFree(pos_d));
    gpuErrchk(cudaFree(end_d));
    gpuErrchk(cudaFree(dir_d));
    gpuErrchk(cudaFree(line_d));
    gpuErrchk(cudaFree(state_d));
    gpuErrchk(cudaFree(schedule_d));
    gpuErrchk(cudaFree(lim1_d));
    gpuErrchk(cudaFree(lim2_d));
    gpuErrchk(cudaFree(costsBE_d));

    gpuErrchk(cudaFree(sums_d));

    // gpuErrchk(cudaFree(victim_flag));
    // gpuErrchk(cudaFree(victim_d));
    // gpuErrchk(cudaFree(length_d));
    // gpuErrchk(cudaFree(sumLength_d));
    gpuErrchk(cudaFree(auxArr));
    gpuErrchk(cudaFree(tmp_arr_d));
    gpuErrchk(cudaFree(todo_d));

    gpuErrchk(cudaFree(toSwap_d));
    gpuErrchk(cudaFree(ivmId_d));
    gpuErrchk(cudaFree(ctrl_d));
    gpuErrchk(cudaFree(nbDecomposed_d));
    gpuErrchk(cudaFree(nbLeaves_d));

    gpuErrchk(cudaStreamDestroy(*stream));
}

void
gpubb::copyH2D()
{
    // printf("%d\n", size);

    int size_m = size * size * nbIVM;
    int size_v = size * nbIVM;
    int size_i = nbIVM;

    gpuErrchk(cudaMemcpy(mat_d, mat_h, size_m * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pos_d, pos_h, size_v * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(end_d, end_h, size_v * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dir_d, dir_h, size_v * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(line_d, line_h, size_i * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(state_d, state_h, size_i * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(costsBE_d, 0, 2 * size_v * sizeof(int)));

    // gpuErrchk(cudaMemcpy(victim_d, victim_h, size_i * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(ctrl_d, ctrl_h, 4 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(nbDecomposed_d, nbDecomposed_h, nbIVM * sizeof(unsigned long long int),
      cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpyToSymbol(nbIVM_d, &nbIVM, sizeof(int)));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // printf("=== copied to device ===\n");
}

void
gpubb::copyH2D_update()
{
    int size_v = size * nbIVM;
    int size_i = nbIVM;

    gpuErrchk(cudaMemcpy(pos_d, pos_h, size_v * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(end_d, end_h, size_v * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(state_d, state_h, size_i * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(line_d, line_h, size_i * sizeof(int), cudaMemcpyHostToDevice));
}

void
gpubb::copyD2H()
{
    int size_i = nbIVM;

    gpuErrchk(cudaMemcpy(ctrl_h, ctrl_d, 4 * sizeof(unsigned int),
      cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(nbDecomposed_h, nbDecomposed_d,
      size_i * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
}
//
//              FLOWSHOP
//
//
// ============================================
#ifdef FSP
void
gpubb::copyH2Dconstant()
{
    gpuErrchk(cudaMemcpyToSymbol(_sumPT, sumPT_h, nbMachines_h * sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_minTempsDep, minTempsDep_h, nbMachines_h * sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_minTempsArr, minTempsArr_h, nbMachines_h * sizeof(int)));

	gpuErrchk(cudaMemcpyToSymbol(_tabJohnson, tabJohnson_h, nbJob_h * somme_h * sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_tempsLag, tempsLag_h, nbJob_h * somme_h * sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_tempsJob, tempsJob_h, nbJob_h * nbMachines_h * sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_machine, machine_h, 2 * somme_h * sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_jobPairs, jobPairs_h, 2 * nbJobPairs_h * sizeof(int)));

    // integer constants
    gpuErrchk(cudaMemcpyToSymbol(nbIVM_d, &nbIVM, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(size_d, &size, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_nbMachines, &nbMachines_h, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_sum, &somme_h, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_boundMode, &arguments::boundMode, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_nbJobPairs, &nbJobPairs_h, sizeof(int)));

    unsigned int initVal=0;
    gpuErrchk(cudaMemcpyToSymbol(_trigger, &initVal, sizeof(unsigned int)));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void
gpubb::initializeBoundFSP()
{
    // get instance data
    (pbb->inst.data)->seekg(0);
    (pbb->inst.data)->clear();

    *(pbb->inst.data) >> nbJob_h;
    *(pbb->inst.data) >> nbMachines_h;

    somme_h = 0;
    for (int i = 1; i < nbMachines_h; i++) somme_h += i;

	nbJobPairs_h = 0;
	for (int i = 1; i < nbJob_h; i++) nbJobPairs_h += i;

    // init bound for GPU
    allocate_host_bound_tmp();

    for (int i = 0; i < nbMachines_h; i++) {
        fillMachine();

        for (int j = 0; j < nbJob_h; j++)
            *(pbb->inst.data) >> tempsJob_h[i * nbJob_h + j];
        fillLag();
        fillTabJohnson();
        fillMinTempsArrDep();
		fillSumPT();
    }

    copyH2Dconstant();
    free_host_bound_tmp();

    // gpuErrchk(cudaMalloc((void **) &front_d, nbIVM * nbMachines_h * sizeof(int)));
    // gpuErrchk(cudaMalloc((void **) &back_d, nbIVM * nbMachines_h * sizeof(int)));


    bd = std::make_unique<gpu_fsp_bound>(nbJob_h,nbMachines_h,nbIVM);
}
#endif /* ifdef FSP */

#ifdef TEST
void
gpubb::initializeBoundTEST()
{
    std::cout<<"init test bound\n";
    size = pbb->inst.size;
    std::cout<<"SIZE : "<<size<<"\n";

    // *(pbb->inst.data) >> size;
    //
    // std::cout<<"SIZE : "<<size<<"\n";

    copyH2DconstantTEST();
}

void
gpubb::copyH2DconstantTEST()
{
    gpuErrchk(cudaMemcpyToSymbol(nbIVM_d, &nbIVM, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(size_d, &size, sizeof(int)));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

#endif

void
gpubb::getStats()
{
    copyD2H();

    int min = INT_MAX;
    int max = 0;
    unsigned long long int gpuDecomposed = 0;

    for (int k = 0; k < nbIVM; k++) {
        min = (nbDecomposed_h[k] < min) ? nbDecomposed_h[k] : min;
        max = (nbDecomposed_h[k] > max) ? nbDecomposed_h[k] : max;
        gpuDecomposed += nbDecomposed_h[k];
    }
    pbb->stats.totDecomposed = gpuDecomposed;

    std::cout<<"Decomposed :\t"<<gpuDecomposed<<"\n";
    printf("Min\t: %d\n", min);
    printf("Max\t: %d\n", max);

    unsigned int nodeCount;
    gpuErrchk(cudaMemcpyFromSymbol(&nodeCount, countNodes_d, sizeof(unsigned int)));

    std::cout<<"decomposed : "<<nodeCount<<"\n";

}

// ==================for DEBUG===================================
void
gpubb::affiche(int M)
{
    cudaMemcpy(state_h, state_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(costsBE_h, costsBE_d, 2 * size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(schedule_h, schedule_d, size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mat_h, mat_d, size * size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_h, pos_d, size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(end_h, end_d, size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dir_h, dir_d, size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nbDecomposed_h, nbDecomposed_d, nbIVM * sizeof(long long int), cudaMemcpyDeviceToHost);
    cudaMemcpy(line_h, line_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(ctrl_h, ctrl_d, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    int ttodo_h;
    gpuErrchk(cudaMemcpyFromSymbol(&ttodo_h, todo, sizeof(unsigned int)));

    for (int i = 0; i < M; i++) {
        if ((state_h[i] != 0) || (i == 0)) {
            printf("\n TODO: %d\t LINE: %d\n", ttodo_h, line_h[i]);
            printf("State: %d\n",state_h[i]);
            //  printf("\t\t PB == %d == %d |||L: %d
            // \n",i,(int)nbDecomposed_h[i],(int)line_h[i]);
            printf("sch\t");
            for (int j = 0; j < size; j++) printf("%2d\t", (int) schedule_h[i * size + j]);
            printf("\n");
            printf("pos\t");
            for (int j = 0; j < size; j++) printf("%2d\t", (int) pos_h[i * size + j]);
            printf("\n");
            printf("end\t");
            for (int j = 0; j < size; j++) printf("%2d\t", (int) end_h[i * size + j]);
            printf("\n");
            printf("dir\t");
            for (int j = 0; j < size; j++) printf("%2d\t", (int) dir_h[i * size + j]);
            printf("\n");
            printf("m[0]\t");
            for (int j = 0; j < size; j++) printf("%2d\t", (int) mat_h[j]);
            printf("\n");
            printf("m[l]\t");
            for (int j = 0; j < size; j++) printf("%2d\t", (int) mat_h[line_h[0] * size + j]);
            printf("\n");
            printf("c[b]\t");
            for (int j = 0; j < size; j++) printf("%2d\t", costsBE_h[i * size * 2 + j]);
            printf("\n");
            printf("c[e]\t");
            for (int j = 0; j < size; j++) printf("%2d\t", costsBE_h[i * size * 2 + size + j]);
            printf("\n");
        }
    }
    printf("END: %d\n", ctrl_h[1]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
} // gpubb::affiche



//=======================================================

//for triggered GPU execution

//=======================================================









bool
gpubb::triggeredNext(int& best, int iter)
{
    bool end = false;

    launchBBkernel(4);
    getExplorationStats(iter,best);

    if (allDone()){
        return true;
    }

	unsigned int target_h=0;
    gpuErrchk(cudaMemcpyFromSymbol(&target_h, targetNode, sizeof(unsigned int)));
    boundLeaves((bool) target_h,best);

    return end;
}






//=======================================================

//for distributed execution only

//=======================================================


void
gpubb::getIntervals(int * pos, int * end, int * ids, unsigned &nb_intervals, const int max_intervals)
{
    memset(pos, 0, max_intervals * size * sizeof(int));
    memset(end, 0, max_intervals * size * sizeof(int));
    // //    fwrk->nb_decomposed = 0;

    if (max_intervals < nbIVM) {
        printf("buffer too small\n");
        exit(-1);
    }

    gpuErrchk(
        cudaMemcpy(pos_h, pos_d, nbIVM * size * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(
        cudaMemcpy(end_h, end_d, nbIVM * size * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(
        cudaMemcpy(state_h, state_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost));
    //copy D2H

    int nbActive = 0;
    for (int k = 0; k < nbIVM; k++) {
        if (state_h[k] != 0) {
            ids[nbActive] = k;
            memcpy(&pos[nbActive*size],&pos_h[k*size],size*sizeof(int));
            memcpy(&end[nbActive*size],&end_h[k*size],size*sizeof(int));

            nbActive++;
        }
    }

    unsigned int nodeCount;
    gpuErrchk(cudaMemcpyFromSymbol(&nodeCount, countNodes_d, sizeof(unsigned int)));
    pbb->stats.totDecomposed = nodeCount;

    nodeCount=0;
    gpuErrchk(cudaMemcpyToSymbol(countNodes_d,&nodeCount,sizeof(unsigned int)));

    nb_intervals = nbActive;
}


void
gpubb::initFromFac(const int nbint, const int* ids, int*pos, int* end)
{
    if (nbint > nbIVM) {
        printf("cannot handle more than %d intervals\n", nbIVM);
        exit(-1);
    }

    if (firstbound) {
        int best = INT_MAX;
        pbb->best_found.getBest(best);

		FILE_LOG(logINFO) << "Init intervals: Bound Root with UB:\t" << best;
		FILE_LOG(logINFO) << "Init intervals: Root:\t" << pbb->best_found;

        // bound root node
        #ifdef FSP
		gpuErrchk( cudaMemcpy(mat_d,pbb->best_found.initial_perm.data(),size*sizeof(int),cudaMemcpyHostToDevice) );

        weakBound(4, best);

        int *d_root_tmp;
        int *d_root_dir_tmp;
        cudaGetSymbolAddress((void **)&d_root_tmp, root_d);
        cudaGetSymbolAddress((void **)&d_root_dir_tmp, root_dir_d);

        gpuErrchk( cudaMemcpy(d_root_tmp, mat_d, size*sizeof(int),cudaMemcpyDeviceToDevice) );
        gpuErrchk( cudaMemcpy(d_root_dir_tmp, dir_d, sizeof(int),cudaMemcpyDeviceToDevice) );
        #endif
        #ifdef TEST
        boundRoot << < 1, 128, sizeof(int) * size, stream[0] >>> (mat_d, dir_d, line_d);
        #endif

        gpuErrchk(cudaMemcpy(costsBE_h,costsBE_d,2*nbIVM*size*sizeof(int),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(dir_h,dir_d,nbIVM*size,cudaMemcpyDeviceToHost));
        firstbound = false;
    }

    const int blocksize = 128;
	dim3 blks((nbIVM * 32 + blocksize) / blocksize);
	setRoot <<< blks, blocksize, 0, stream[0] >> > (mat_d, dir_d);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    memset(pos_h, 0, nbIVM * size * sizeof(int));
    memset(end_h, 0, nbIVM * size * sizeof(int));
    memset(state_h, 0, nbIVM * sizeof(int));
    memset(line_h, 0, nbIVM * sizeof(int));

    for (int k = 0; k < nbint; k++) {
        int id = ids[k];

        if (id >= nbIVM) {
			FILE_LOG(logERROR) << "ID > nbIVMs!";
            exit(-1);
        }

        int l = 0;
        // while ((posVect[ind2D(k, l)] == pos[l]) && l < line[k]) l++;
        line_h[id] = l;

        memcpy(pos_h + id*size, pos + k * size, size * sizeof(int));
        memcpy(end_h + id*size, end + k * size, size * sizeof(int));

        state_h[id] = -1;
    }

    unsigned int nodeCount=0;
    gpuErrchk(cudaMemcpyToSymbol(countNodes_d,&nodeCount,sizeof(unsigned int)));

    copyH2D_update();
}

int gpubb::getDeepSubproblem(int *ret, const int N){
    cudaMemcpy(state_h, state_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(line_h, line_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(schedule_h, schedule_d, nbIVM * size * sizeof(int), cudaMemcpyDeviceToHost);

    int NN=4;

	int *cmax;
	cudaMallocManaged(&cmax,nbIVM*sizeof(int));
	// searchSolutions<<<(nbIVM+NN-1) / NN, NN * 32, smem, stream>>>(schedule_d, makespans, best);

	int gbest;
	int *bestsol;
	cudaMallocManaged(&bestsol,size*sizeof(int));
    pbb->best_found.getBestSolution(bestsol,gbest);

    struct timespec startt,endt;
    clock_gettime(CLOCK_MONOTONIC,&startt);

    // int smem = (NN * (3*size + nbMachines_h)) * sizeof(int);
	// xOver_makespans<32><<<(nbIVM+NN-1) / NN, NN * 32, smem, stream>>>(schedule_d, cmax, state_d, bestsol, lim1_d, lim2_d);
#ifdef FSP
    int smem = (NN * (size + nbMachines_h)) * sizeof(int);
    makespans<32><<<(nbIVM+NN-1) / NN, NN * 32, smem, stream[0]>>>
        (schedule_d, cmax, state_d);
#endif

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
	cudaFree(bestsol);

    clock_gettime(CLOCK_MONOTONIC,&endt);

    cudaMemcpy(schedule_h, schedule_d, nbIVM * size * sizeof(int), cudaMemcpyDeviceToHost);

	FILE_LOG(logDEBUG) << "Evaluated Solutions [ "<< (endt.tv_sec-startt.tv_sec)+(endt.tv_nsec-startt.tv_nsec)/1e9<<" ]";

    std::vector<std::tuple<int,int>>v;
    for(int i=0;i<nbIVM;i++)
	{
		// (*i)->ub=ls->localSearch((*i)->schedule, (*i)->limit1+1, (*i)->limit2);

        if(state_h[i]>0){
            // v.push_back(std::make_tuple(i,size-line_h[i]));
            v.push_back(std::make_tuple(i,cmax[i]));
        }
    }
    std::sort(begin(v), end(v),
        [](std::tuple<int, int> const &t1, std::tuple<int, int> const &t2) {
            return std::get<1>(t1) < std::get<1>(t2); // sort according to second (cost)
        }
    );

    int nb=std::min(v.size(),(size_t)N);

    int count=0;
    for(auto i:v)
    {
        if(count>=nb)break;

        int id=std::get<0>(i);
        if(state_h[id])
        {
            for(int k=0;k<size;k++){
                ret[count*size+k]=schedule_h[id*size+k];
            }
            // for(int k=0;k<size;k++){
            //     printf("%2d,",ret[count*size+k]);
            // }
            // printf("\n");

            count++;
        }
    }

    cudaFree(cmax);

    return count;
}
