#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <memory>

#include "pbab.h"
//solution,stats,instance
#include "arguments.h"
//nbivms_gpu, problem, boundMode, branchingMode, findAll, printSolutions,ws_strategy

#include "subproblem.h"
#include "solution.h"
#include "ttime.h"
#include "log.h"

#include "libbounds.h"

#include "gpubb.h"

// all CUDA from this file
#include "./bb_kernels.cu"

gpubb::gpubb(pbab * _pbb)
{
    pbb  = _pbb;
    size = pbb->size;
    nbIVM    = arguments::nbivms_gpu;
    ringsize = nbIVM;

    // CPU-based bound (simple) : needed for solution eval on CPU
    // as in 'ivm_bound' constructor (should make a function...)
    if (arguments::problem[0] == 'f') {
        bound_fsp_weak *bd=new bound_fsp_weak();
        bd->init(pbb->instance);
        // bd->set_instance(pbb->instance);
        // bd->init();
        bd->branchingMode=arguments::branchingMode;
        bound=bd;
    }
    if (arguments::problem[0] == 't' || arguments::problem[0] == 'n') {
        bound_nqueens *bd=new bound_nqueens();
        bd->set_instance(pbb->instance);
        bd->init();
        bd->branchingMode=arguments::branchingMode;
        bound=bd;
    }

    initialUB = INT_MAX;
    pbb->sltn->getBest(initialUB);

	FILE_LOG(logINFO) << "GPU with nbIVM:\t" << nbIVM;
	FILE_LOG(logINFO) << "Initial UB:\t" << initialUB;
    gpuErrchk(cudaStreamCreate(&stream));
    gpuErrchk(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    pthread_mutex_init(&mutex_end,NULL);

    pthread_mutex_lock(&mutex_end);
    allEnd = false;
    pthread_mutex_unlock(&mutex_end);

    // "one time events"
    startclock = true;
    firstbound = true;

    search_cut = 1.0;

	execmode.triggered = false;

	// executionmode.triggered=false;

}

gpubb::~gpubb()
{
    free_on_device();
    // free etc
}

//
void
gpubb::initialize()
{
    setHypercubeConfig(); //work stealing
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
        pbb->sltn->getBest(best);
		FILE_LOG(logINFO) << "Init Full : Bound Root with UB:\t" << best;
		FILE_LOG(logINFO) << "Init Full : Root :\t" << *(pbb->root_sltn);

		int *bestsol_d;
		gpuErrchk( cudaMalloc(&bestsol_d,size*sizeof(int)) );
		gpuErrchk( cudaMemcpy(bestsol_d,pbb->root_sltn->perm,size*sizeof(int),cudaMemcpyHostToDevice) );

        // bound root node
        #ifdef FSP
        boundRoot << < 1, 1024, (nbMachines_h + sizeof(int)) * size >>> (mat_d, dir_d, line_d, costsBE_d, sums_d, bestsol_d, best, arguments::branchingMode);
        #endif
        #ifdef TEST
        boundRoot << < 1, 128, sizeof(int) * size >>> (mat_d, dir_d, line_d);
        #endif
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        cudaFree(bestsol_d);

        gpuErrchk(cudaMemcpy(costsBE_h,costsBE_d,2*nbIVM*size*sizeof(int),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(dir_h,dir_d,nbIVM*size,cudaMemcpyDeviceToHost));

        firstbound = false;
    }

	dim3 blks((nbIVM * 32 + 127) / 128);
	setRoot << < blks, 128, 0, stream >> > (mat_d, dir_d);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    memset(pos_h, 0, nbIVM * size * sizeof(int));
    memset(end_h, 0, nbIVM * size * sizeof(int));
    memset(state_h, 0, nbIVM * sizeof(int));
    memset(line_h, 0, nbIVM * sizeof(int));

	for(int i=0;i<size;i++){
		end_h[i]=size-i-1;
	}
	state_h[0] = -1;

    copyH2D_update();
    // affiche(1);
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
    // goToNext_dense<<< (nbIVM+127) / 128, 128, 0, stream >>>(mat_d, pos_d, end_d, dir_d, line_d, state_d, nbDecomposed_d, counter_d, NN);

	//wide mapping : one warp = one IVM
    // assume:
    // 1 block = NN warps = NN IVM
    size_t smem = NN * (2 * size * sizeof(int) + 2 * sizeof(int));
    goToNext2<4><<< (nbIVM+NN-1) / NN, NN * 32, smem, stream >>>(mat_d, pos_d, end_d, dir_d, line_d, state_d, nbDecomposed_d, counter_d);

#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
}

void
gpubb::launchBBkernel(const int NN)
{
    int best = INT_MAX;

    pbb->sltn->getBest(best);

    gpuErrchk(cudaMemset(counter_d, 0, 6 * sizeof(unsigned int)));
    gpuErrchk(cudaMemset(flagLeaf, 0, nbIVM * sizeof(int)));
    unsigned int target_h = 0;
    gpuErrchk(cudaMemcpyToSymbol(_trigger, &target_h, sizeof(unsigned int)));
    gpuErrchk(cudaMemcpyToSymbol(targetNode, &target_h, sizeof(unsigned int)));
    gpuErrchk(cudaMemset(costsBE_d, 999999, 2 * size * nbIVM * sizeof(int)));

    size_t smem = NN * (3*nbMachines_h + 3 * size + 2) * sizeof(int);
    multistep_triggered<4><<< (nbIVM+NN-1) / NN, NN * 32, smem, stream >>>
    (mat_d, pos_d, end_d, dir_d, line_d, state_d, nbDecomposed_d, counter_d, schedule_d, lim1_d, lim2_d,costsBE_d,flagLeaf, best, initialUB);

#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
    // affiche(1);
}


bool
gpubb::allDone()
{
    // gpuErrchk(cudaMemcpy(state_h,state_d,nbIVM*sizeof(int),cudaMemcpyDeviceToHost));

    unsigned int end_h = 0;
    gpuErrchk(cudaMemcpyToSymbol(deviceEnd, &end_h, sizeof(unsigned int)));

    // blockReduce+atomic
    if (nbIVM >= 1024)
        checkEnd << < (nbIVM + 1023) / 1024, 1024, 0, stream >>> (state_d);
    else if (nbIVM == 512)
        checkEnd << < (nbIVM + 511) / 512, 512, 0, stream >> > (state_d);
    else{
        checkEnd << < (2*nbIVM - 1) / nbIVM, std::max(32,nbIVM), 0, stream >> > (state_d);
    }
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    gpuErrchk(cudaMemcpyFromSymbol(&end_h, deviceEnd, sizeof(unsigned int)));
    return (end_h==0);
}

void
gpubb::buildMapping()
{
    gpuErrchk(cudaMemset(auxArr, 0, 256 * sizeof(int)));

    dim3 blocksz(256);
    dim3 numbblocks((nbIVM + blocksz.x - 1) / blocksz.x);
    size_t smem = 2 * blocksz.x * sizeof(int);
    reduce<<< numbblocks, blocksz, smem, stream >> > (todo_d, tmp_arr_d, auxArr, blocksz.x);
	reduce2<<< numbblocks, blocksz, 0, stream >> > (tmp_arr_d, auxArr);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

	switch(arguments::boundMode){
		case 1:
		{
			prepareBound2<<<(nbIVM+PERBLOCK-1) / PERBLOCK, 32 * PERBLOCK, 0, stream>>>(lim1_d, lim2_d, todo_d, ivmId_d, toSwap_d, tmp_arr_d);
			break;
		}
		case 2:
		{
	    	prepareBound<<<(nbIVM+127) / 128, 128, 0, stream>>>(schedule_d, costsBE_d,dir_d,line_d,lim1_d, lim2_d, todo_d, ivmId_d, toSwap_d, tmp_arr_d);
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

    pbb->sltn->getBest(best);

    int nbsteals = 0;
	localFoundNew=false;

    while (true) {
        nbsteals += steal_in_device(iter);

        if(execmode.triggered){
            //perform whole BB in single kernel..break if a threshold of empty explorers is reached.
            end = triggeredNext(best, iter);
        }else{
            //one BB step
            end = next(best, iter);
        }

        //conditions to trigger communication with master
        if(!arguments::singleNode){
            if(((nbsteals > (nbIVM/5)) && iter>100) || localFoundNew || pbb->ttm->period_passed(WORKER_BALANCING)){
                break;
            }
        }
        if(end){
            break;
        }
    }

	//return true if allEnd
    return end;
}

// returns true iff no more work available
bool
gpubb::next(int& best, int& iter)
{
    if (startclock) {
        clock_gettime(CLOCK_REALTIME, &starttime);
        startclock = false;
    }
    bool end = false;

    selectAndBranch(4);
    getExplorationStats(iter,best);

    if (allDone()){
        // printf("all done\n");
        return true;
    }

    bool reachedLeaf=decode(4);

    //if not 'strong-bound-only' ...
	if(arguments::boundMode != 1){
		reachedLeaf = weakBound(4, best);
	}

	unsigned int target_h=0;
    gpuErrchk(cudaMemcpyFromSymbol(&target_h, targetNode, sizeof(unsigned int)));
    reachedLeaf = (bool) target_h;

    boundLeaves(reachedLeaf,best);


    if(arguments::boundMode != 0){
        buildMapping();

        cudaMemset(sums_d, 0, 2 * nbIVM * sizeof(int));
        cudaMemset(costsBE_d, 0, 2 * size * nbIVM * sizeof(int));

        int ttodo_h;
        gpuErrchk(cudaMemcpyFromSymbol(&ttodo_h, todo, sizeof(unsigned int)));
        dim3 boundblocks;

        switch(arguments::boundMode){
            case 1://no pre-bounding...
            {
                boundblocks.x = (2 * ttodo_h+127) / 128;
#ifdef FSP
                boundJohnson<<<boundblocks, 128, nbJob_h * nbMachines_h + 64 * nbJob_h, stream>>>(schedule_d, lim1_d, lim2_d, line_d, costsBE_d, sums_d, state_d, toSwap_d,ivmId_d, nbLeaves_d, ctrl_d, flagLeaf, best);
#endif
#ifdef TEST
                /*
                boundStrong kernel here!!! (if strong bound only)
                */
#endif
                sortedPrune <<< (nbIVM+PERBLOCK-1) / PERBLOCK, 32 * PERBLOCK, 0, stream >>> (mat_d, dir_d, line_d, costsBE_d, sums_d, state_d, flagLeaf, best);
                break;
            }
            case 2:
            {
                if(ttodo_h>0){
                    boundblocks.x = (ttodo_h+127) / 128;
#ifdef FSP
                    boundOne <<< boundblocks, 128, nbJob_h * nbMachines_h,stream >>>
                    (schedule_d, lim1_d, lim2_d, dir_d, line_d, costsBE_d, toSwap_d, ivmId_d,best, front_d, back_d);
#endif
#ifdef TEST
                    /*
                    TO IMPLEMENT
                    boundStrong kernel here!!! (if mixed bound)
                    */
                    boundStrongFront <<< boundblocks, 128, 0, stream >>>
                    (schedule_d, lim1_d, lim2_d, line_d, toSwap_d, ivmId_d, costsBE_d);
#endif
                    prune2noSort <<< (nbIVM+PERBLOCK-1) / PERBLOCK, 32 * PERBLOCK, 0, stream >> > (mat_d, dir_d, line_d, costsBE_d, state_d, best);
                }
                break;
    	    }
        }
    }
    #ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #endif

    iter++;

    return end; // (ctrl_h[gpuEnd] == 1);
} // gpubb::next



bool gpubb::decode(const int NN)
{
    unsigned int target_h = 0;

    // cudaMemcpy(lim1_h, lim1_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(lim2_h, lim2_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
	// cudaMemcpy(schedule_h, schedule_d, size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < nbIVM; i++) {
    //     printf("%d %d\t",lim1_h[i],lim2_h[i]);
    //     for (int j = 0; j < size; j++) printf("%d\t", (int) schedule_h[i * size + j]);
    //     printf("\n");
	// }
	// printf("----- \n\n");

	switch(arguments::boundMode){
        case 1: //strong bound
        {
            gpuErrchk(cudaMemset(todo_d, 0, nbIVM * sizeof(int)));
            gpuErrchk(cudaMemset(flagLeaf, 0, nbIVM * sizeof(int)));
            gpuErrchk(cudaMemcpyToSymbol(targetNode, &target_h, sizeof(unsigned int)));

            decodeIVMandFlagLeaf <<< nbIVM / 4, 128, 4 * 3 * sizeof(int), stream >>>
                (mat_d, dir_d, pos_d, lim1_d, lim2_d, line_d, schedule_d, state_d, todo_d, flagLeaf);

            gpuErrchk(cudaMemcpyFromSymbol(&target_h, targetNode, sizeof(unsigned int)));
     	    break;
        }
        case 0: //weak bound
        case 2: //mixed
        {
            size_t smem = (NN * (size + 3)) * sizeof(int);
            decodeIVM<<< (nbIVM+NN-1) / NN, NN * 32, smem, stream >>>
                (mat_d, dir_d, pos_d, lim1_d, lim2_d, line_d, schedule_d, state_d);

            break;
        }
    }

    // cudaMemcpy(lim1_h, lim1_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(lim2_h, lim2_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
	// cudaMemcpy(schedule_h, schedule_d, size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
	// for (int i = 0; i < nbIVM; i++) {
    //     printf("%d %d\t",lim1_h[i],lim2_h[i]);
	// 	for (int j = 0; j < size; j++) printf("%d\t", (int) schedule_h[i * size + j]);
	// 	printf("\n");
	// }
	// printf("====== \n\n");

    return ((bool) target_h);
}

bool
gpubb::weakBound(const int NN, const int best)
{
    gpuErrchk(cudaMemset(flagLeaf, 0, nbIVM * sizeof(int)));

    unsigned int target_h = 0;
    gpuErrchk(cudaMemcpyToSymbol(targetNode, &target_h, sizeof(unsigned int)));

    // cudaMemcpy(costsBE_h, costsBE_d, 2 * size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i=0;i<nbIVM;i++){
    //     for(int j=0;j<size;j++)
    //         printf("%4d ",costsBE_h[2*i*size+j]);
    //     printf("\n");
    //     for(int j=0;j<size;j++)
    //         printf("%4d ",costsBE_h[(2*i+1)*size+j]);
    //     printf("\n");
    // }
    // printf("-------- \n");

    gpuErrchk(cudaMemset(costsBE_d, 999999, 2 * size * nbIVM * sizeof(int)));
	size_t smem;

#ifdef FSP
    smem = (NN * (size + 3 * nbMachines_h)) * sizeof(int);
    if(arguments::branchingMode>0){
        boundWeak_BeginEnd<32><<<(nbIVM+NN-1) / NN, NN * 32, smem, stream >>>(lim1_d, lim2_d, line_d, schedule_d, costsBE_d, state_d, front_d, back_d, best, flagLeaf);
    }else if(arguments::branchingMode==-1){
        boundWeak_Begin<32> << < (nbIVM+NN-1) / NN, NN * 32, smem, stream >> >
        (lim1_d, lim2_d, line_d, schedule_d, costsBE_d, state_d, front_d, back_d, best, flagLeaf);
    }
#endif
#ifdef TEST
    //shared memory for schedules
    smem = NN * size * sizeof(int);
    boundWeakFront<<<(nbIVM+NN-1) / NN, NN * 32, smem, stream >>>(state_d,schedule_d,lim1_d,lim2_d,line_d,costsBE_d,flagLeaf);
#endif

#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    // cudaMemcpy(costsBE_h, costsBE_d, 2 * size * nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i=0;i<nbIVM;i++){
    //     for(int j=0;j<size;j++)
    //         printf("%4d ",costsBE_h[2*i*size+j]);
    //     printf("\n");
    //     for(int j=0;j<size;j++)
    //         printf("%4d ",costsBE_h[(2*i+1)*size+j]);
    //     printf("\n");
    // }
    // printf("====== \n");



    gpuErrchk(cudaMemset(todo_d, 0, nbIVM * sizeof(int)));
    if(arguments::branchingMode>0){
        smem = (NN * (2*size + 2) * sizeof(int));
        chooseBranchingSortAndPrune<<< (nbIVM+NN-1) / NN, NN * 32, smem, stream >> >
        (mat_d, dir_d, pos_d, lim1_d, lim2_d, line_d, schedule_d, costsBE_d, prio_d, state_d, todo_d, best, initialUB,arguments::branchingMode);
    }
    else if(arguments::branchingMode==-1){
        smem = (NN * (2*size + 2) * sizeof(int));
        ForwardBranchSortAndPrune<<< (nbIVM+NN-1) / NN, NN * 32, smem, stream >> >
        (mat_d, dir_d, pos_d, lim1_d, lim2_d, line_d, schedule_d, costsBE_d, prio_d, state_d,
            todo_d, best, flagLeaf);
    }
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    gpuErrchk(cudaMemcpyFromSymbol(&target_h, targetNode, sizeof(unsigned int)));
    return ((bool) target_h);
} // gpubb::decodeAndBound

static int gpuverb=0;
void gpubb::getExplorationStats(const int iter, const int best)
{
    // Don't remove this memcopy : used in work stealing
    gpuErrchk(cudaMemcpy(counter_h, counter_d, 6 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //just to get some feedback
    if ((gpuverb==1) && (iter > 0)) {
    // if ((gpuverb==1) && (iter % 1000 == 0) && (iter > 0)) {
		FILE_LOG(logINFO) << "Expl: " << counter_h[exploringState] << "\tEmpty: " << counter_h[emptyState];
    }
}

bool
gpubb::boundLeaves(bool reached, int& best)
{
    if(!reached)return false;

    int flags[nbIVM];

    cudaMemcpy(flags, flagLeaf, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);

    // cudaMemcpy(line_h, line_d, nbIVM * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(schedule_h, schedule_d, nbIVM * size * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(pos_h, pos_d, nbIVM * size * sizeof(int), cudaMemcpyDeviceToHost);

    bool newUB = false;

    for (int k = 0; k < nbIVM; ++k) {
        if (flags[k] == 1) {
            int cost=bound->evalSolution(schedule_h+k*size);
            pbb->stats.leaves++;
            FILE_LOG(logINFO) << "Evaluated Leaf\t" << cost;

            bool update;
            if(arguments::findAll)update=(cost<=best);
            else update=(cost<best);

            if (update) {
                best = cost;
                pbb->sltn->update(schedule_h+k*size,cost);

				localFoundNew = true;

                pbb->foundSolution=true;

				// std::cout << "Worker found " << cost << "\n";
                //print new best solution
                FILE_LOG(logINFO) << "Worker found " << cost;
                FILE_LOG(logINFO) << *(pbb->sltn);

                newUB = true;
            }
        }
    }

    return newUB;
}


int
gpubb::steal_in_device(int iter)
{
    // if (counter_h[exploringState] >= nbIVM-2048) return 0;
    if (counter_h[exploringState] >= 75*nbIVM/100) return 0;

    struct timespec startt,endt;
    clock_gettime(CLOCK_MONOTONIC,&startt);

    adapt_workstealing(2, nbIVM / 8);

    computeLength << < nbIVM / PERBLOCK, 32 * PERBLOCK, 0, stream >> >
    (pos_d, end_d, length_d, state_d, sumLength_d);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    //    search_cut = 0.1;
    computeMeanLength << < (nbIVM + 127) / 128, 128, 0, stream >> >
    (sumLength_d, meanLength_d, search_cut, nbIVM); // (int)(ctrl_h[2]+1));
#ifndef NDEDUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    int nbdims = topoDimensions;
    int dimb = iter % nbdims;
    int from, to, dim, q;

    for (int s = dimb; s < dimb + nbdims; s++) {
        dim  = s % nbdims;
        q    = (1 << topoB[dim]);
        from = iter & (q - 1);
        to   = from + q;

        for (int off = from; off < to; off++)
            prepareShare << < (nbIVM + 127) / 128, 128, 0, stream >> >
            (state_d, victim_flag, victim_d, length_d, meanLength_d, off & (q - 1), topoB[dim], topoA[dim]);
    }
#ifndef NDEDUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    share_on_gpu2 <<< nbIVM / PERBLOCK, 32 * PERBLOCK, 0, stream >>>
    (mat_d, pos_d, end_d, dir_d, line_d, 1, ws_granular, state_d, victim_flag, victim_d, ctrl_d);
#ifndef NDEDUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    unsigned int tmp = 0;
    gpuErrchk(cudaMemcpyFromSymbol(&tmp, gpuBalancedIntern, sizeof(unsigned int)));

    clock_gettime(CLOCK_MONOTONIC,&endt);
	FILE_LOG(logDEBUG1) << "GPU load balanced. Steals: "<<tmp<<" ["<< (endt.tv_sec-startt.tv_sec)+(endt.tv_nsec-startt.tv_nsec)/1e9<<" ]";

    return tmp;
} // steal_in_device

void
gpubb::adapt_workstealing(int min, int max)
{
    if ((counter_h[exploringState] >= (int) (7 * nbIVM / 10)) && (search_cut < 0.8))
        search_cut += 0.1;
    else if ((counter_h[exploringState] < (int) (7 * nbIVM / 10)) && (search_cut > 0.2))
        search_cut -= 0.1;
    ws_granular = 2;

	FILE_LOG(logDEBUG1) << "GPUWS length coefficient: "<<search_cut;
}

void
gpubb::setHypercubeConfig()
{
    //printf("setting up topology %d\n", nbIVM);
    switch (nbIVM) {
        case 1:
            topoDimensions = 1;
            topoA[0]       = 0;
            topoB[0]       = 0;
            break;

        case 4:
            topoDimensions = 1;
            topoA[0]       = 0;
            topoB[0]       = 2;
            break;

        case 64: // 64 == 4*4*4
            topoDimensions = 3;
            topoA[0]       = 0; topoA[1]       = 2; topoA[2]       = 4;
            topoB[0]       = 2; topoB[1]       = 2; topoB[2]       = 2;
            break;

        case 128: // 64 == 4*4*4
            topoDimensions = 3;
            topoA[0]       = 0;            topoA[1]       = 2;            topoA[2]       = 4;
            topoB[0]       = 2;            topoB[1]       = 2;            topoB[2]       = 3;
            break;

        case 512: // 512 == 4*4*4*8
        {
            topoDimensions = 4;
            for(int i=0;i<topoDimensions;i++){
                topoA[i]=2*i;
                topoB[i]=2;
            }
            topoB[3] = 3;
            break;
        }
        case 1024: // 2**(2 2 2 2 2)
        {
            topoDimensions = 5;
            for(int i=0;i<topoDimensions;i++){
                topoA[i]=2*i;
                topoB[i]=2;
            }
            break;
        }
        case 2048: // 8*8*8*4 == 2**(3+3+3+2)
        {
            topoDimensions = 5;
            for(int i=0;i<topoDimensions;i++){
                topoA[i]=2*i;
                topoB[i]=2;
            }
            topoB[4] = 3;
            break;
        }
        case 4096: // 4096 == 8**4 (2**12)
        {
            topoDimensions = 6;
            for(int i=0;i<topoDimensions;i++){
                topoA[i]=2*i;
                topoB[i]=2;
            }
            break;
        }
        case 8192: // 4096 == 16 * 8**3 (2**13)
        {
            topoDimensions = 6;
            for(int i=0;i<topoDimensions;i++){
                topoA[i]=2*i;
                topoB[i]=2;
            }
            topoB[5]       = 3;
            break;
        }
        case 16384: // (2**14)
        {
            topoDimensions = 7;
            for(int i=0;i<topoDimensions;i++){
                topoA[i]=2*i;
                topoB[i]=2;
            }
            break;
        }
        case 32768: // 32768 == 8**5
        {
            topoDimensions = 7;
            for(int i=0;i<topoDimensions;i++){
                topoA[i]=2*i;
                topoB[i]=2;
            }
            topoB[6]       = 3;
            break;
        }
        case 65536: // 65536 == 2 * 8**5 (2**16)
        {
            topoDimensions = 8;
            for(int i=0;i<topoDimensions;i++){
                topoA[i]=2*i;
                topoB[i]=2;
            }
            break;
        }
        case 131072: // 2**17
        {
            topoDimensions = 8;
            for(int i=0;i<topoDimensions;i++){
                topoA[i]=2*i;
                topoB[i]=2;
            }
            topoB[7]       = 3;
            break;
        }

        default:
            printf("invalid ivm-nb\n");
            exit(0);
            break;
    }
} // gpubb::setHypercubeConfig

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
	FILE_LOG(logINFO) << "size: " << size << "nbIVM: " << nbIVM;

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

    victim_h     = (int *) calloc(size_i, sizeof(int));
    length_h     = (int *) calloc(size_v, sizeof(int));
    ctrl_h       = (unsigned int *) calloc(8, sizeof(unsigned int));
    counter_h    = (unsigned int *) calloc(6, sizeof(unsigned int));
    meanLength_h = (int *) calloc(size, sizeof(int));
    costsBE_h    = (int *) calloc(2 * size_v, sizeof(int));

	depth_histo_h = (unsigned int*)calloc(size, sizeof(unsigned int));

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

    //  gpuErrchk(cudaMalloc((void **)&split_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &victim_flag, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &victim_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &length_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &sumLength_d, size * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &meanLength_d, size * sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &depth_histo_d, size * sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &flagLeaf, size_i * sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &auxArr, 256 * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &auxEnd, 256 * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &tmp_arr_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &todo_d, size_i * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &toSwap_d, size_v * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &ivmId_d, size_v * sizeof(int)));

    gpuErrchk(cudaMalloc((void **) &ctrl_d, 4 * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **) &counter_d, 6 * sizeof(unsigned int)));

    gpuErrchk(cudaMalloc((void **) &nbDecomposed_d, size_i * sizeof(unsigned long long int)));
    gpuErrchk(cudaMalloc((void **) &nbLeaves_d, size_i * sizeof(int)));

    // set to 0
    gpuErrchk(cudaMemset(counter_d, 0, 6 * sizeof(unsigned int)));
    gpuErrchk(cudaMemset(sumLength_d, 0, size * sizeof(int)));
    gpuErrchk(cudaMemset(victim_flag, 0, size_i * sizeof(int)));
    gpuErrchk(cudaMemset(ivmId_d, 0, size_v * sizeof(int)));
    gpuErrchk(cudaMemset(sums_d, 0, 2 * size_i * sizeof(int)));
    unsigned int zero=0;
    gpuErrchk(cudaMemcpyToSymbol(countNodes_d,&zero,sizeof(unsigned int)));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
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

    gpuErrchk(cudaFree(victim_flag));
    gpuErrchk(cudaFree(victim_d));
    gpuErrchk(cudaFree(length_d));
    gpuErrchk(cudaFree(sumLength_d));
    gpuErrchk(cudaFree(auxArr));
    gpuErrchk(cudaFree(tmp_arr_d));
    gpuErrchk(cudaFree(todo_d));

    gpuErrchk(cudaFree(toSwap_d));
    gpuErrchk(cudaFree(ivmId_d));
    gpuErrchk(cudaFree(ctrl_d));
    gpuErrchk(cudaFree(nbDecomposed_d));
    gpuErrchk(cudaFree(nbLeaves_d));

    gpuErrchk(cudaStreamDestroy(stream));
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

    gpuErrchk(cudaMemcpy(victim_d, victim_h, size_i * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(ctrl_d, ctrl_h, 4 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(nbDecomposed_d, nbDecomposed_h, nbIVM * sizeof(unsigned long long int),
      cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(depth_histo_d, depth_histo_h, size * sizeof(unsigned int), cudaMemcpyHostToDevice));

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
    gpuErrchk(cudaMemcpyToSymbol(_nbMachines, &nbMachines_h, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_sum, &somme_h, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(size_d, &size, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_nbJob, &size, sizeof(int)));
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
    (pbb->instance->data)->seekg(0);
    (pbb->instance->data)->clear();

    *(pbb->instance->data) >> nbJob_h;
    *(pbb->instance->data) >> nbMachines_h;

    somme_h = 0;
    for (int i = 1; i < nbMachines_h; i++) somme_h += i;

	nbJobPairs_h = 0;
	for (int i = 1; i < nbJob_h; i++) nbJobPairs_h += i;

    // init bound for GPU
    allocate_host_bound_tmp();

    for (int i = 0; i < nbMachines_h; i++) {
        fillMachine();

        for (int j = 0; j < nbJob_h; j++)
            *(pbb->instance->data) >> tempsJob_h[i * nbJob_h + j];
        fillLag();
        fillTabJohnson();
        fillMinTempsArrDep();
		fillSumPT();
    }

    copyH2Dconstant();
    free_host_bound_tmp();

    gpuErrchk(cudaMalloc((void **) &front_d, nbIVM * nbMachines_h * sizeof(int)));
    gpuErrchk(cudaMalloc((void **) &back_d, nbIVM * nbMachines_h * sizeof(int)));
}
#endif /* ifdef FSP */

#ifdef TEST
void
gpubb::initializeBoundTEST()
{
    *(pbb->instance->data) >> size;
    copyH2DconstantTEST();
}

void
gpubb::copyH2DconstantTEST()
{
    gpuErrchk(cudaMemcpyToSymbol(nbIVM_d, &nbIVM, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(size_d, &size, sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(_nbJob, &size, sizeof(int)));

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

    // printf("Min\t: %d\n", min);
    // printf("Max\t: %d\n", max);
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

    for (int i = 0; i < M; i++) {
        if ((state_h[i] != 0) || (i == 0)) {
            printf("\t TODO: %d\t%d\n", ctrl_h[0], line_h[i]);

            printf("state %d\n",state_h[i]);
            //  printf("\t\t PB == %d == %d |||L: %d
            // \n",i,(int)nbDecomposed_h[i],(int)line_h[i]);
            printf("sch\t");
            for (int j = 0; j < size; j++) printf("%d\t", (int) schedule_h[i * size + j]);
            printf("\n");
            printf("pos\t");
            for (int j = 0; j < size; j++) printf("%d\t", (int) pos_h[i * size + j]);
            printf("\n");
            printf("end\t");
            for (int j = 0; j < size; j++) printf("%d\t", (int) end_h[i * size + j]);
            printf("\n");
            printf("dir\t");
            for (int j = 0; j < size; j++) printf("%d\t", (int) dir_h[i * size + j]);
            printf("\n");
            printf("m[0]\t");
            for (int j = 0; j < size; j++) printf("%d\t", (int) mat_h[j]);
            printf("\n");
            printf("m[l]\t");
            for (int j = 0; j < size; j++) printf("%d\t", (int) mat_h[line_h[0] * size + j]);
            printf("\n");
            printf("c[b]\t");
            for (int j = 0; j < size; j++) printf("%d\t", costsBE_h[i * size * 2 + j]);
            printf("\n");
            printf("c[e]\t");
            for (int j = 0; j < size; j++) printf("%d\t", costsBE_h[i * size * 2 + size + j]);
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
gpubb::triggeredNext(int& best, int& iter)
{
    if (startclock) {
        clock_gettime(CLOCK_REALTIME, &starttime);
        startclock = false;
    }
    bool end = false;

    launchBBkernel(4);
    getExplorationStats(iter,best);

    if (allDone()){
        return true;
    }

	unsigned int target_h=0;
    gpuErrchk(cudaMemcpyFromSymbol(&target_h, targetNode, sizeof(unsigned int)));
    boundLeaves((bool) target_h,best);

    iter++;

    return end;
}






//=======================================================

//for distributed execution only

//=======================================================


void
gpubb::getIntervals(int * pos, int * end, int * ids, int &nb_intervals, const int max_intervals)
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
            // states[nbActive] = matrices[k]->getState();

            memcpy(&pos[nbActive*size],&pos_h[k*size],size*sizeof(int));
            memcpy(&end[nbActive*size],&end_h[k*size],size*sizeof(int));
            // for(int i=0;i<size;i++){
            //     pos[nbActive * size + i] = pos_h[k*size + i];
            //     end[nbActive * size + i] = end_h[k*size + i];
            // }

    //         matrices[k]->getInterval(&pos[nbActive * size], &end[nbActive * size]);
            nbActive++;
        }
	//
	//
    //     //        printf("%d",matrices[k]->decomposedNodes);fflush(stdout);
    //     // fwrk->nb_decomposed += matrices[k]->decomposedNodes;
    //     // matrices[k]->decomposedNodes = 0;
    //     // pthread_mutex_unlock(&matrices[k]->mutex_ivm);
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
        pbb->sltn->getBest(best);

		FILE_LOG(logINFO) << "Init intervals: Bound Root with UB:\t" << best;
		FILE_LOG(logINFO) << "Init intervals: Root:\t" << *(pbb->root_sltn);

		int *bestsol_d;
		gpuErrchk( cudaMalloc(&bestsol_d,size*sizeof(int)) );
		gpuErrchk( cudaMemcpy(bestsol_d,pbb->root_sltn->perm,size*sizeof(int),cudaMemcpyHostToDevice) );

        // bound root node
        #ifdef FSP
        boundRoot <<< 1, 1024, (nbMachines_h + sizeof(int)) * size >>> (mat_d, dir_d, line_d, costsBE_d, sums_d, bestsol_d, best, arguments::branchingMode);
        #endif
        #ifdef TEST
        boundRoot << < 1, 128, sizeof(int) * size >>> (mat_d, dir_d, line_d);
        #endif
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(bestsol_d);

        gpuErrchk(cudaMemcpy(costsBE_h,costsBE_d,2*nbIVM*size*sizeof(int),cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(dir_h,dir_d,nbIVM*size,cudaMemcpyDeviceToHost));
        firstbound = false;
    }

	dim3 blks((nbIVM * 32 + 127) / 128);
	setRoot <<< blks, 128, 0, stream >> > (mat_d, dir_d);
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
    pbb->sltn->getBestSolution(bestsol,gbest);

    struct timespec startt,endt;
    clock_gettime(CLOCK_MONOTONIC,&startt);

    // int smem = (NN * (3*size + nbMachines_h)) * sizeof(int);
	// xOver_makespans<32><<<(nbIVM+NN-1) / NN, NN * 32, smem, stream>>>(schedule_d, cmax, state_d, bestsol, lim1_d, lim2_d);
    int smem = (NN * (size + nbMachines_h)) * sizeof(int);
    makespans<32><<<(nbIVM+NN-1) / NN, NN * 32, smem, stream>>>
        (schedule_d, cmax, state_d);
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
