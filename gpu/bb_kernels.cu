#define PERBLOCK 4 // warps per block
#define TILE_SZ 32 // tile size

#define MAXJOBS 800


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <assert.h>

// don't change order of following includes! (I should set up separate compilation...)
// GPU:common
#include "../include/gpu_helper.cuh"
#include "../include/reductions.cuh"
#include "../include/gpu_ivm_navi.cuh"

// GPU:bounds
#ifdef FSP
#include "gpu_fsp_bound.cuh"
#endif /* ifdef FSP */
#ifdef TEST
# include "gpu_test_bound.cuh"
#endif /* ifdef TEST */

#include "gpu_interval.cuh"
#include "loadbalance.cuh"

#include "selection_kernels.cu"



#ifdef FSP
template<unsigned NN>
__global__ void
multistep_triggered(int * jobMats_d, int * posVecs_d, int * endVecs_d, int * dirVecs_d, int * line_d, int * state_d, unsigned long long int * count, unsigned int * counter_d, int *schedule_d, int* lim1_d, int*lim2_d, int*costsBE_d, int *flagLeaf, const int best,const int initialUB)
{
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / g.size(); // global ivm id
    int thPos = g.thread_rank();
    int warpID = threadIdx.x / g.size();

    //shared memory...... pos,end,state,line
    extern __shared__ int shar[];
    int *front = (int*)&shar;//partial schedule begin
    int *back    = (int *)&front[NN * _nbMachines];  // partial schedule end[M]
    int *remain  = (int *)&back[NN * _nbMachines];   // remaining work[M]
    // int * pv = shar;
    int * sched = (int *)&remain[NN * _nbMachines];
    // int * pv = (int *)&remain[NN * _nbMachines];
    int * pv = (int *)&sched[NN * size_d];
    int * ev  = (int *) &pv[NN * size_d];
    int * state = (int *) &ev[NN * size_d];
    int * line  = (int *) &state[NN];

    __shared__ int lim1[NN];
    __shared__ int lim2[NN];

    front += warpID*_nbMachines;
    back += warpID*_nbMachines;
    remain += warpID*_nbMachines;

    // int * pv   = posv + warpID * size_d;            // shared mem - vector
    sched += warpID * size_d;
    pv += warpID * size_d;
    ev += warpID * size_d;

    //load to shared memory ...
    for (int i = g.thread_rank(); i < size_d; i+=warpSize){
        pv[i] = posVecs_d[ivm * size_d + i];
        ev[i] = endVecs_d[ivm * size_d + i];
    }
    if (thPos == 0) {
        state[warpID] = state_d[ivm];
        line[warpID]  = line_d[ivm];
    }

    g.sync();
    //=====================================
    // (just to compute less indices)
    int * jm   = jobMats_d + ivm * size_d * size_d; // global mem matrix

    //fixed max number of steps...
    for(int i=0;i<500;i++){
        // initializing IVM
        if (state[warpID] < 0){
            (void)initStep(g,jm,pv,dirVecs_d + ivm * size_d,line[warpID],state[warpID]);
        }
        g.sync();
        if (state[warpID] > 0) {
            exploreStep(g,jm,pv,ev,line[warpID],state[warpID]);
            g.sync();
            if (state[warpID] == 1) {
                count[ivm]++;//per IVM counter
                for (int i = g.thread_rank(); i< size_d; i += g.size()){
                    assert(line[warpID] < size_d - 1);
                    assert(line[warpID] >= 0);
                    parallelGoDown(jm, pv, line[warpID], i);
                    pv[line[warpID]] = 0;
                }
                g.sync();
                if (g.thread_rank() == 0)
                    line[warpID]++;
            }
        }
        g.sync();

        if(state[warpID]!=0){
            tile_decodeIVM(g, jm, pv, &dirVecs_d[ivm*size_d],line[warpID],lim1[warpID], lim2[warpID], sched);

            g.sync();
            tile_resetRemain(g, remain);
            tile_scheduleFront(g, sched, lim1[warpID], _tempsJob, front, remain);
            tile_scheduleBack(g, sched, lim2[warpID], _tempsJob, back, remain);
            g.sync();

            tile_addFrontAndBound(g,back,front,remain,&sched[lim1[warpID]+1],size_d-line[warpID],&costsBE_d[2 * ivm * size_d]);
            tile_addBackAndBound(g,back,front,remain,&sched[lim1[warpID]+1],size_d-line[warpID],&costsBE_d[(2 * ivm + 1) * size_d]);
            //
            int *jmrow = jm+line[warpID]*size_d;
            g.sync();

            if(g.thread_rank()==0){
                if (line[warpID] == size_d - 1) {
                    flagLeaf[ivm] = 1;
                    atomicInc(&targetNode, UINT_MAX);
                    jmrow[0] = negative_d(jmrow[0]);
                }
            }
            g.sync();

            if(flagLeaf[ivm]){
                if(g.thread_rank()==0)atomicInc(&_trigger,INT_MAX);
                break;
            }

            int dir=tile_MinBranch(g, jmrow, &costsBE_d[2 * ivm * size_d], &dirVecs_d[ivm*size_d], line[warpID],initialUB);
            dir=g.shfl(dir,0);
            g.sync();//!!!! every thread has dir

            if(thPos==0){
                if(line[warpID]==size_d-1){
                    jmrow[0] = negative_d(jmrow[0]);
                }
                //reverse
                int i1=0;
                int i2=size_d - line[warpID]-1;
                if(dirVecs_d[ivm*size_d+line[warpID]-1]!=dir){
                    while(i1<i2)
                    {
                        swap_d(&jmrow[i1],&jmrow[i2]);
                        i1++; i2--;
                    }
                    i1=lim1[warpID]+1;
                    i2=lim2[warpID]-1;
                    while(i1<i2)
                    {
                        swap_d(&sched[i1],&sched[i2]);
                        i1++; i2--;
                    }
                }
            }

            g.sync();
            tile_prune(g, jmrow, costsBE_d+2*ivm*size_d, dir, line[warpID], best);
        }else{
            if(g.thread_rank()==0)atomicInc(&_trigger,INT_MAX);
            break;
        }
        if(_trigger>2*nbIVM_d/10){
            break;
        }
    }

    // increment statistics counters
    if (g.thread_rank() == 0) {
        if (state[warpID] > 0) atomicInc(&counter_d[exploringState], INT_MAX);
        if (state[warpID] == 0) atomicInc(&counter_d[emptyState], INT_MAX);
        if (state[warpID] < 0) atomicInc(&counter_d[initState], nbIVM_d);
    }
    g.sync();

    // back to global mem
    for (int i = thPos; i < size_d; i+=g.size()) {
        posVecs_d[ivm * size_d + i] = pv[i];
    }
    state_d[ivm] = state[warpID];
    line_d[ivm]  = line[warpID];
}
#endif

/*decode IVMs using one warp (thread_block_tile) per IVM

- decode operation is partially parallelized.
- resulting schedules in shared mem
*/
__global__ void // __launch_bounds__(128, 16)
decodeIVM(const int *jobMats_d,const int *dirVecs_d,const int *posVecs_d,int *limit1s_d,int *limit2s_d,const int *line_d,int *schedules_d, const int *state_d)
{
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / tile32.size(); // global ivm id
    int warpID = threadIdx.x / tile32.size();

    // SHARED MEMORY
    extern __shared__ int smemDecode[];
    int *prmu   = smemDecode;
    int *l1      = (int *)&prmu[4 * size_d];
    int *l2      = (int *)&l1[4];

    prmu += warpID * size_d;

    int line=line_d[ivm];
    const int *jm = jobMats_d + ivm * size_d * size_d;

    // nothing to do
    if (state_d[ivm] == 0) return;

    tile_decodeIVM(tile32, jm, &posVecs_d[ivm*size_d],&dirVecs_d[ivm*size_d],line, l1[warpID], l2[warpID], prmu);
    tile32.sync();

    //back to main mem
    for (int i = tile32.thread_rank(); i < size_d; i+=tile32.size()) {
        schedules_d[index2D(i,ivm)]=prmu[i];
    }
    limit1s_d[ivm] = l1[warpID];
    limit2s_d[ivm] = l2[warpID];
} // prepareSchedules

__global__ void
flagLeaf_fillTodo(int *flagLeaf, int *todo_d, const int *l1, const int *l2, const int *line_d, const int *state_d){
    int ivm = blockIdx.x*blockDim.x + threadIdx.x;

    if (line_d[ivm] == size_d - 1) {
        flagLeaf[ivm] = 1;
        atomicInc(&targetNode, UINT_MAX);
    }

    todo_d[ivm] = 0;
    if (state_d[ivm] != 0) {
        todo_d[ivm] = l2[ivm] - l1[ivm] - 1;
    }
}

__global__ void
chooseBranchingSortAndPrune(int *jobMats_d,int *dirVecs_d,const int *posVecs_d,int *limit1s_d,int *limit2s_d, const int *line_d,int *schedules_d,int *costsBE_d,int *prio_d, int *state_d,int *todo_d,const int best,const int initialUB,const int branchStrategy)
{
    auto tile32 = tiled_partition<32>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // global ivm id
    int warpID = threadIdx.x / warpSize;
    int thPos = threadIdx.x % warpSize;

    // SHARED MEMORY
    extern __shared__ bool smemPrune[];
    int *jmrow = (int *)&smemPrune;   // schedule[N]
    int *prio      = (int *)&jmrow[4 * size_d];
    int *line    = (int *)&prio[4 * size_d];//&l2[4];

    jmrow += warpID * size_d;

    //load schedule limits and line to smem
    if (thPos == 0) {
        line[warpID] = line_d[ivm];
//        sum[warpID]  = 0;//999999;
    }
    tile32.sync();

    for (int i = tile32.thread_rank(); i < size_d; i+=warpSize) {
        jmrow[i]=jobMats_d[index3D(line[warpID],i,ivm)];
    }
    tile32.sync();

    // nothing to do
    if (state_d[ivm] == 0) return;

    int dir;
    switch(branchStrategy){
    case 1:
    {
        dir=tile_branchMaxSum<32>(tile32, jmrow, &costsBE_d[2 * ivm * size_d], line[warpID]);
        break;
    }
    case 2:
    {
        dir=tile_MinBranch<32>(tile32, jmrow, &costsBE_d[2 * ivm * size_d], &dirVecs_d[ivm*size_d], line[warpID],initialUB);
        break;
    }
    case 3:
    {
        dir=tile_branchMinMin<32>(tile32, jmrow, &costsBE_d[2 * ivm * size_d], &dirVecs_d[ivm*size_d], line[warpID]);
        break;
    }
    }

    //only thread 0 has correct value : broadcast
    dir=tile32.shfl(dir,0);
    tile32.sync();//!!!! every thread has dir

    //order jobs in next row
    if(thPos==0){
        dirVecs_d[ivm*size_d + line[warpID]] = dir; //tile32.shfl(dir,0);
        assert(dirVecs_d[ivm*size_d + line[warpID]] == dir);
        // dirVecs_d[ivm*size_d] = dir;

        if(line[warpID]==size_d-1){
            jmrow[0] = negative_d(jmrow[0]);
        }

        //reverse
        int i1=0;
        int i2=size_d - line[warpID]-1;

        int prev_dir=(line[warpID]>0)?dirVecs_d[ivm*size_d+line[warpID]-1]:0;

        if(prev_dir!=dir){
            while(i1<i2)
            {
                swap_d(&jmrow[i1],&jmrow[i2]);
                i1++;
                i2--;
            }
        }
        if(prev_dir==1 && dir==0){
            for (int l = 0; l < size_d - line[warpID]; l++){
                schedules_d[ivm*size_d+limit1s_d[ivm]+1+l] = absolute_d(jmrow[l]);
            }
        }
    }

    tile32.sync();

    //prune
    tile_prune<32>(tile32, jmrow, costsBE_d+2*ivm*size_d, dir, line[warpID], best);

    //prapare strong bound
    if(_boundMode>=1){
        if (thPos == 0) {
            //popc ballot ... !
            todo_d[ivm] = 0;
            for(int i=0;i<size_d - line[warpID];i++){
                if(jmrow[i] >= 0){
                    // printf("job=%d\n",jmrow[i]);
                    todo_d[ivm]++;//count non-pruned
                }
            }

            // while (jmrow[todo_d[ivm]] >= 0 && todo_d[ivm] < size_d - line[warpID])
            //     todo_d[ivm]++;//count non-pruned
            // printf("todo[%d]=%d\n",ivm,todo_d[ivm]);
        }
    }

    //back to main mem
    __syncthreads();
    for (int i = thPos; i < size_d; i+=warpSize) {
        jobMats_d[index3D(line[warpID],i,ivm)]=jmrow[i];
    }
}


template < typename T >
__global__ void // __launch_bounds__(128, 16)
ForwardBranchSortAndPrune(T *jobMats_d,T *dirVecs_d,const T *posVecs_d,T *limit1s_d,T *limit2s_d,
const T *line_d,T *schedules_d,int *costsBE_d,int *prio_d, T *state_d,int *todo_d,const int best,int *flagLeaf)
{
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // global ivm id
    int warpID = threadIdx.x / warpSize;
    int thPos = threadIdx.x % warpSize;

    // SHARED MEMORY
    extern __shared__ bool smemPrune[];
    int *jmrow = (int *)&smemPrune;   // schedule[N]
    int *prio      = (int *)&jmrow[4 * size_d];
    int *line    = (int *)&prio[4 * size_d];//&l2[4];
//    int *sum     = (int *)&line[4];

    jmrow += warpID * size_d;
    int i;

    //load schedule limits and line to smem
    if (thPos == 0) {
        line[warpID] = line_d[ivm];
    }

    for (i = 0; i <= (size_d / warpSize); i++) {
        if (i * warpSize + thPos < size_d) {
            jmrow[i * warpSize + thPos]=jobMats_d[index3D(line[warpID],i * warpSize + thPos,ivm)];
        }
    }
    tile32.sync();

    // nothing to do
    if (state_d[ivm] == 0) return;

    int dir=0; //tile_chooseBranching(tile32, jmrow, &costsBE_d[2 * ivm * size_d], &dirVecs_d[ivm*size_d], line[warpID]);
    dir=tile32.shfl(dir,0);

    int job;
    for (i = 0; i <= (size_d / warpSize); i++) {
        if (i * warpSize + thPos < size_d - line[warpID]) {
            job = jmrow[i * warpSize + thPos];
            prio_d[ivm * size_d + job]=costsBE_d[(2 * ivm + (dir)) * size_d + job];
        }
    }

    if (thPos == 0) {
        // setting directionVector
        dirVecs_d[index2D(line[warpID], ivm)] = dir; // (p0>p1);

        //COMMENT OUT TO DISABLE SORTING
        // this is gnome sort (tested it: on small arrays better than insert,
        // quick, std::, bubble, selection sort !) O(n^2) worst case time, O(1)
        // gnomeSortSequential(jmrow,prio+warpID*size_d,1, size_d - line[warpID]);
    }

    if(thPos==0){
        if(line[warpID]==size_d-1){
            jmrow[0] = negative_d(jmrow[0]);
        }
    }

    tile_prune(tile32, jmrow, costsBE_d+2*ivm*size_d, dir, line[warpID], best);

    if (thPos == 0) {
        todo_d[ivm] = 0;
        while (jmrow[todo_d[ivm]] >= 0 && todo_d[ivm] < size_d - line[warpID]) todo_d[ivm]++;//count non-pruned
    }

    __syncthreads();
    for (i = 0; i <= (size_d / warpSize); i++) {
        if (i * warpSize + thPos < size_d) {
            jobMats_d[index3D(line[warpID],i * warpSize + thPos,ivm)]=jmrow[i * warpSize + thPos];
            //=jobMats_d[index3D(line[warpID],i * warpSize + thPos,ivm)];
        }
    }
}



template < typename T >
__global__ void
prune2noSort(T *jobMats_d, const T *dirVecs_d, const T *line_d,
       const int *costsBE_d, const T *state_d, const int best)
{
    // thread indexing
    int ivm   = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // global
                                                                    // ivm id
    int thPos = threadIdx.x % warpSize;

    int dir = dirVecs_d[index2D(line_d[ivm], ivm)];
    int l   = 0;

    // pruning
    if (state_d[ivm] != 0) {
        for (l = 0; l <= size_d / warpSize; l++) {
            if (l * warpSize + thPos < size_d - line_d[ivm]) {
                int job = jobMats_d[index3D(line_d[ivm], l * warpSize + thPos, ivm)];

                if (job < 0) continue;                            // already pruned in phase I

                int val = costsBE_d[index2D(job, 2 * ivm + dir)]; // LB
#ifdef FINDALL
                if (val > best) {                                 // find ALL optimal solutions
#else  /* ifdef FINDALL */
                if (val >= best) {
#endif /* ifdef FINDALL */
                    jobMats_d[index3D(line_d[ivm], l * warpSize + thPos, ivm)] = negative_d(job); // eliminate node !
                }
            }
        }// for warp
    } // if(state)
}

//

/***********************************************************************/
template < typename T >
__global__ void prune(T *jobMats_d, T *dirVecs_d, const T *line_d,
                      const int *costsBE_d, const  int *sums_d, const T *state_d, unsigned int *ctrl_d, unsigned int *counter_d, int best) {
    /**** thread indexing ***********/
    int ivm   = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // global
                                                                    // ivm id
    int thPos = threadIdx.x % warpSize;                             // threadId
                                                                    // within
                                                                    // IVM

    //  int best = (int)ctrl_d[4];

    // setting directionVector
    if (thPos == 0) {
        if (sums_d[2 * ivm] > sums_d[2 * ivm + 1]) {
            dirVecs_d[index2D(line_d[ivm], ivm)] = 0;
        } else {
            dirVecs_d[index2D(line_d[ivm], ivm)] = 1;
        }
    }

    __syncthreads();

    // pruning
    int l = 0;

    if (state_d[ivm] != 0) {
        for (l = 0; l <= size_d / warpSize; l++) {
            if (l * warpSize + thPos < size_d - line_d[ivm]) {
                int job = absolute_d(jobMats_d[index3D(line_d[ivm], l * warpSize + thPos, ivm)]);
                int val = (1 - dirVecs_d[index2D(line_d[ivm], ivm)]) *
                          costsBE_d[index2D(job, 2 * ivm)] +
                          dirVecs_d[index2D(line_d[ivm], ivm)] *
                          costsBE_d[index2D(job, 2 * ivm + 1)];

                if (val >= best) {
                    jobMats_d[index3D(line_d[ivm], l * warpSize + thPos, ivm)] = negative_d(
                        jobMats_d[index3D(line_d[ivm], l * warpSize + thPos, ivm)]); //
                                                                                     // eliminate
                                                                                     // node
                                                                                     // !
                }
            }
        }// for warp
    }

    if (threadIdx.x == 0) {
        counter_d[exploringState] = 0;
        counter_d[emptyState]     = 0;
        counter_d[initState]      = 0;
    }
}

template < typename T >
__global__ void sortedPrune(T *jobMats_d, T *dirVecs_d, const T *line_d,
                            const int *costsBE_d, const  int *sums_d, const T *state_d, int *flagLeaf, const int best) {
    /**** thread indexing ***********/
    int ivm   = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // global
                                                                    // ivm id
    int thPos = threadIdx.x % warpSize;                             // threadId
                                                                    // within
                                                                    // IVM
    int dir;

    // setting directionVector
    if (sums_d[2 * ivm] >= sums_d[2 * ivm + 1]) {
        dir                                  = 0;
        dirVecs_d[index2D(line_d[ivm], ivm)] = 0;
    } else {
        dir                                  = 1;
        dirVecs_d[index2D(line_d[ivm], ivm)] = 1;
    }

    __syncthreads();

    // SORTING (insert sort)
    // int l=0;
    if (state_d[ivm] != 0) {
        // for(l=0; l<=size_d/warpSize; l++){
        if (thPos == 0) {
            int i, j, key;

            for (i = 1; i < size_d - line_d[ivm]; i++) {       // for(i=1;i<N;i++)
                key = jobMats_d[index3D(line_d[ivm], i, ivm)]; // arr[i];
                j   = i;

                // while(j>0 && arr[j-1] > key)
                while (j > 0 && costsBE_d[index2D(jobMats_d[index3D(line_d[ivm], j - 1, ivm)], 2 * ivm + dir)] > costsBE_d[index2D(key, 2 * ivm + dir)]) {
                    jobMats_d[index3D(line_d[ivm], j, ivm)] = jobMats_d[index3D(line_d[ivm], j - 1, ivm)]; //
                                                                                                           // arr[j]=arr[j-1];
                    j--;
                }
                jobMats_d[index3D(line_d[ivm], j, ivm)] = key;
            }
        }
    }

    __syncthreads();

    int l = 0;

    // int jobs[MAXJOBS];

    //  int job = absolute(jobMats_d[index3D(line_d[ivm], l*warpSize + thPos,
    // ivm)]);
    // __syncthreads();

    // pruning
    if (state_d[ivm] != 0) {
        for (l = 0; l <= size_d / warpSize; l++) {
            if (l * warpSize + thPos < size_d - line_d[ivm]) {
                //        jobs[thPos] = absolute(jobMats_d[index3D(line_d[ivm],
                // l*warpSize + thPos, ivm)]);
                int job = absolute_d(jobMats_d[index3D(line_d[ivm], l * warpSize + thPos, ivm)]);

                // int val = costsBE_d[index2D(jobs[thPos], 2 * ivm + dir)];
                int val = costsBE_d[index2D(job, 2 * ivm + dir)];
#ifdef FINDALL
                if (val > best) {
#else
                if (val >= best) {
#endif

                    //  jobMats_d[index3D(line_d[ivm], l*warpSize + thPos, ivm)]
                    // = negative(jobs[thPos]);
                    jobMats_d[index3D(line_d[ivm], l * warpSize + thPos, ivm)] = negative_d(job); //
                                                                                                  // eliminate
                                                                                                  // node
                                                                                                  // !
                }
            }
        }// for warp
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
__global__ void prepareBound(int* schedule_d, int* costsBE_d, int* dirVecs_d, int* line_d, int *limit1s_d, int *limit2s_d, int *todo_d, int *ivmId_d, int *toSwap_d, int *tempArr_d, int *state_d, int best)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ivm   = tid;
    int job;

    if (state_d[ivm] == 0) return;

    assert(limit2s_d[ivm]-limit1s_d[ivm]-1 == size_d-line_d[ivm]);

    int cc=0;
    int dir=dirVecs_d[index2D(line_d[ivm],ivm)];
    for(int i=limit1s_d[ivm] + 1;i<limit2s_d[ivm];i++){
        job=schedule_d[index2D(i,ivm)];
        if(costsBE_d[index2D(job,2*ivm+dir)]<best){
            ivmId_d[tempArr_d[ivm] + cc]  = ivm;
            toSwap_d[tempArr_d[ivm] + cc] = i;//limit1s_d[ivm] + 1 + i;
            cc++;
        }
    }

    if (tid == 0) {
        todo = tempArr_d[nbIVM_d - 1] + todo_d[nbIVM_d - 1];
    }
}


template < typename T >
__global__ void
prepareBound2(T *limit1s_d, T *limit2s_d, int *todo_d, int *ivmId_d, int *toSwap_d, int *tempArr_d, int *state_d)
{
    int thPos = threadIdx.x % warpSize;
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;

    // int ivm = blockIdx.x * PERBLOCK + threadIdx.x / warpSize;
    int ivm = tid / warpSize;

    if (state_d[ivm] == 0) return;

    if(thPos==0){
        int cc=0;
        for(int i=limit1s_d[ivm] + 1;i<limit2s_d[ivm];i++){
            ivmId_d[tempArr_d[ivm] + cc]  = ivm;
            toSwap_d[tempArr_d[ivm] + cc] = i;//limit1s_d[ivm] + 1 + i;
            cc++;
        }
    }

    //    0 ... l2 - l1 - 2
    // l1+1 ... l2 - 1



    // for(l = thPos; l < todo_d[ivm]; l += warpSize)
    // {
    //     ivmId_d[tempArr_d[ivm] + l]  = ivm; // ivm;
    //     toSwap_d[tempArr_d[ivm] + l] = limit1s_d[ivm] + 1 + l;
    // }

    // for (l = 0; l <= size_d / warpSize; l++) {
    //     if (l * warpSize + thPos < todo_d[ivm]) {
    //         ivmId_d[tempArr_d[ivm] + l * warpSize + thPos]  = ivm; // ivm;
    //         toSwap_d[tempArr_d[ivm] + l * warpSize + thPos] = limit1s_d[ivm] + 1 + l * warpSize + thPos;
    //     }
    // }

    if (tid == 0) {
        todo = tempArr_d[nbIVM_d - 1] + todo_d[nbIVM_d - 1];
    }
}

template < typename T >
__global__ void
share_inter_gpu2(int den, const T *mat, const T *pos, T *end,
                 const T *dir, const T *line, const T *state, T *steal_mat, T *steal_pos, T *steal_end,
                 T *steal_dir, T *steal_line,
                 T *steal_state, int toDivide)
{
    int thPos = threadIdx.x % warpSize;
    int ivm   = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    int i = 0;
    int l = 0;

    if (ivm < toDivide) {
        // IF thief empty && victim exploring && victim interval large enough...
        if ((steal_state[ivm] == 0) && (state[ivm] == 1)) {
            // ... THEN share
            while (pos[index2D(l, ivm)] == end[index2D(l, ivm)] && l < line[ivm] && l < size_d - 3) l++;

            if (pos[index2D(l, ivm)] >= end[index2D(l, ivm)]) {
                for (int k = 0; k <= size_d / warpSize; k++) {
                    if (k * warpSize + thPos < size_d) {
                        steal_pos[index2D(k * warpSize + thPos, ivm)] = size_d - (k * warpSize + thPos) - 1;
                        steal_end[index2D(k * warpSize + thPos, ivm)] = size_d - (k * warpSize + thPos) - 1;
                    }
                }

                if (thPos == 0) {
                    steal_pos[index2D(thPos, ivm)] = size_d;
                    steal_state[ivm]               = 0;
                }
                steal_state[ivm] = 0;
            } else {
                for (int k = 0; k <= size_d / warpSize; k++) {
                    if (k * warpSize + thPos < l) {
                        steal_pos[index2D(k * warpSize + thPos, ivm)] = pos[index2D(k * warpSize + thPos, ivm)];
                        steal_dir[index2D(k * warpSize + thPos, ivm)] = dir[index2D(k * warpSize + thPos, ivm)];
                    }

                    for (i = 0; i < l; i++) {
                        if (k * warpSize + thPos <
                            size_d) steal_mat[index3D(i, k * warpSize + thPos,
                                                      ivm)] = mat[index3D(i, k * warpSize + thPos, ivm)];
                    }

                    if (k * warpSize + thPos < size_d) {
                        steal_end[index2D(k * warpSize + thPos, ivm)]    = end[index2D(k * warpSize + thPos, ivm)];
                        steal_mat[index3D(l, k * warpSize + thPos, ivm)] = mat[index3D(l, k * warpSize + thPos, ivm)];
                    }
                }

                if (thPos == 0) {
                    steal_dir[index2D(l, ivm)] = dir[index2D(l, ivm)];
                    steal_pos[index2D(l, ivm)] = cuttingPosition(l, den, pos + ivm * size_d, end + ivm * size_d,
                                                                 mat + ivm * size_d * size_d);
                    end[index2D(l, ivm)] = steal_pos[index2D(l, ivm)] - 1;

                    for (i = l + 1; i < size_d; i++) {
                        steal_pos[index2D(i, ivm)] = 0;
                        end[index2D(i, ivm)]       = size_d - i - 1;
                    }
                    steal_line[ivm]  = l;
                    steal_state[ivm] = 1;
                }
            }
        }
    }
} // share_inter_gpu2

template < typename T >
__global__ void setRoot(T *mat, T *dir) {
    int thPos = threadIdx.x % warpSize;
    int ivm   = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    for (int l = thPos; l < size_d; l+=warpSize) {
        mat[index3D(0, l, ivm)] = root_d[l];
    }

    if (thPos == 0) dir[index2D(0, ivm)] = root_dir_d;
}

//
//
//
//          FLOWSHOP
//
//
#ifdef FSP
template < typename T >
__global__ void __launch_bounds__(128, 8) boundJohnson(const T * schedules_d, const T * limit1s_d, const T * limit2s_d, const T * line_d, int * costsBE_d, int * sums_d, const T * state_d, const int * toSwap_d, const int * ivmId_d, unsigned int * bdleaves_d, unsigned int * ctrl_d, int * flagLeaf, const int best) {
    /**** thread indexing ****/
    register int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    register int BE    = tid & 1;
    register int ivmnb = ivmId_d[(tid >> 1)]; // the ivm tid is working on

    /***** shared memory declarations *****/
    extern __shared__ unsigned char sharedArray[];
    unsigned char *tempsJob_sh = (unsigned char *)sharedArray;
    char *permut_sh            = (char *)&tempsJob_sh[_nbMachines * size_d];

    if (threadIdx.x < size_d) {
        for (int j = 0; j < _nbMachines; j++) tempsJob_sh[j * size_d + threadIdx.x] =
                (unsigned char)_tempsJob[j * size_d + threadIdx.x];
    }

    //  if (tid < 2 * ctrl_d[toDo]) {
    if (tid < 2 * todo) {
        if (tid % 2 == 0) {
            for (int i = 0; i < size_d; i++) permut_sh[index2D(i, threadIdx.x >> 1)] =
                    schedules_d[index2D(i, ivmnb)];
        }
    }

    __syncthreads();

    /*******************************************/

    //  if (tid < 2 * ctrl_d[toDo]) {
    if (tid < 2 * todo) {
        char limit1 = limit1s_d[ivmnb] + 1 - BE;
        char limit2 = limit2s_d[ivmnb] - BE;

        char Swap1 = toSwap_d[(tid >> 1)];
        char Swap2 = (1 - BE) * limit1 + BE * limit2;

        char jobnb = permut_sh[index2D(Swap1, threadIdx.x >> 1)];

        int where = ivmnb * 2 * size_d + BE * size_d + (int)jobnb;

        if (line_d[ivmnb] < (size_d - 1)) { // boundNodes
            costsBE_d[where] = computeCost(permut_sh, Swap1, Swap2, limit1, limit2, _tempsJob, threadIdx.x >> 1, _tabJohnson, best);

            //      costsBE_d[where] = computeCost(permut_sh, Swap1, Swap2,
            // limit1, limit2, tempsJob_sh, threadIdx.x >> 1, _tabJohnson);// +
            // BE * limit2;
            atomicAdd(&sums_d[2 * ivmnb + BE], costsBE_d[where]);
        } else if (BE == 0) { // boundLeaves
            //      char pos = posVecs_d[index2D(line_d[ivmnb], ivmnb)];
            //      jobMats_d[index3D(line_d[ivmnb], pos, ivmnb)] =
            //          negative(jobMats_d[index3D(line_d[ivmnb], pos, ivmnb)]);
            if (state_d[ivmnb] == 1) bdleaves_d[ivmnb]++;

            flagLeaf[ivmnb] = 1;
            atomicInc(&ctrl_d[foundLeaf], UINT_MAX);
        }
    }
}
#endif /* ifdef FSP */
