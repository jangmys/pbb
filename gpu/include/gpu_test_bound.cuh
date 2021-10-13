/*


    Bound 1 : for example constraint checking ...


 */
template < typename T >
__device__ void computeBoundsWeakTile(thread_block_tile<TILE_SZ> g, const T *prmu, const T l1, const T l2, const T line,int* costs){
    int job;

	//size_d - line = #remaining jobs
	//prmu[l1+1] = first unscheduled job

	for (int i = g.thread_rank(); i<size_d-line; i += warpSize){
        job = prmu[l1+1+i]; // each thread grabs one job
		//...and computes bound for subproblem obtained by swapping "job" and prmu[l1+1]
		//...
		//...
		//...
		//...
		costs[job]=0; // implement (weak) bounding function here !
        //
	}
}

template < typename T >
__global__ void boundWeakFront(const T* state_d,const T* schedules_d,const T* lim1_d,const T* lim2_d,const T* line_d,int *costsBE_d,int *flagLeaf){
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // global ivm id
    int warpID = threadIdx.x / warpSize;

    T line=line_d[ivm];
    T l1=lim1_d[ivm];
    T l2=lim2_d[ivm];

    extern __shared__ bool sharedTESTweak[];
    T *prmu = (int *)&sharedTESTweak;

    int i;

    prmu += warpID * size_d;
    for (i = 0; i <= (size_d / warpSize); i++) {
        if (i * warpSize + tile32.thread_rank() < size_d) {
            prmu[i * warpSize + tile32.thread_rank()]=schedules_d[ivm*size_d+i * warpSize + tile32.thread_rank()];
        }
    }
    tile32.sync();

    // nothing to do
    if (state_d[ivm] == 0) return;

	//one warp to compute bounds for children of (prmu,l1,l2)
	//
	//
	// TO IMPLEMENT ...
	//
	//
    computeBoundsWeakTile(tile32, prmu,l1,l2,line,&costsBE_d[2 * ivm * size_d]);

	// Leaf is reached ... flag to trigger copy to CPU
    if(tile32.thread_rank()==0){
        if (line == size_d - 1) {
            flagLeaf[ivm] = 1;
            atomicInc(&targetNode, UINT_MAX);
        }
    }
}

/*


 Bound 2: stronger one...


 */


//single threaded function ...
//
template <typename T>
__device__ int computeBoundStrongThread(const T* permutation, const int limit1, const int limit2, const int toSwap1, const int toSwap2)
{
    //this can be a "copy" of the "bornes_calculer" function implemented for multi-core in bound_****.cpp

    //everything is const to avoid duplicating permutation in memory!!!!
    //swap1 and swap2 are used to avoid actually doing swap(&permutation[toSwap1],&permutation[toSwap2]) ...
    //example :

    int lb=0;

    int job;
    //loop over jobs scheduled in beginning
    for (int j = 0; j <= limit1; j++) {
        if (j == toSwap1)
            job = permutation[toSwap2];
        else if (j == toSwap2)
            job = permutation[toSwap1];
        else
            job = permutation[j];

            /*
                do something here
             */

        lb += 0; //put some partial cost incurred by "job"
    }

    //loop over jobs not yet fixed...
    //
    //only fixing jobs in front, so (limit2 == size_d)
    for (int j = limit1 + 1; j < limit2; j++) {
        if (j == toSwap1)
            job = permutation[toSwap2];
        else if (j == toSwap2)
            job = permutation[toSwap1];
        else
            job = permutation[j];

            /*
            do something here
            */

        lb += 0; //now add some underestimation of the cost incurred by unsched
    }

    return lb;
}

// todo == number of bounds to compute
// toSwap_d and ivmID_d contain mapping of threads to subproblems ...
template <typename T>
__global__ void
boundStrongFront(const T * schedules_d, const T * lim1_d, const T * lim2_d, const T * line_d,  const int * toSwap_d, const int * ivmId_d, int * costsBE_d)
{
    // thread indexing
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < todo) {
        int ivmnb = ivmId_d[tid];                       // use mapping to get the ivm, thread tid is working on
        int limit1 = lim1_d[ivmnb] + 1;                 // get limits of partial schedule
        int limit2 = lim2_d[ivmnb];

        int Swap1 = toSwap_d[tid];  //from mapping, get index of job to place...
        int Swap2 = limit1;//...and where to place it (only front)

        int jobnb = schedules_d[index2D(Swap1, ivmnb)];
        int where = ivmnb * 2 * _nbJob + (int) jobnb; //index where to write LB in cost array ...

        if (line_d[ivmnb] < (_nbJob - 1)) { // compute bounds!!!!!!!
            costsBE_d[where] = computeBoundStrongThread(schedules_d+ivmnb*size_d, limit1, limit2, Swap1, Swap2);
        }
    }
}

//if no bounds computed at root simply keep this
//otherwise compute bounds for root decomposition, prune and store first row of mat in root_d (root_dir_d only needed if begin/end branching)
template < typename T >
__global__ void boundRoot(T *mat, T *dir, T *line) {
    int thPos = threadIdx.x % warpSize;
    int ivm   = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if (ivm == 0) {
        for(int l=thPos; l<size_d; l+=warpSize){
		    mat[index3D(0, l, ivm)] = l;
            root_d[l] = l;
        }
        root_dir_d = 0;
    }
}
