__global__ void
goToNext_dense2(int * jobMats_d, int * posVecs_d, int * endVecs_d, int * dirVecs_d, int * line_d, int * state_d, unsigned long long int * count, unsigned int * counter_d, int NN)
{

}

/*
 * count : count decomposed nodes
 * counter_d : count current states
 */
__global__ void
goToNext_dense(int * jobMats_d, int * posVecs_d, int * endVecs_d, int * dirVecs_d, int * line_d, int * state_d, unsigned long long int * count, unsigned int * counter_d, int NN)
{
    int ivm = (blockIdx.x * blockDim.x + threadIdx.x);

    int state = state_d[ivm];
    int line  = line_d[ivm];

    int * pv   = &posVecs_d[ivm * size_d];
    int * ev   = &endVecs_d[ivm * size_d];
    int * jm   = jobMats_d + ivm * size_d * size_d; // global mem matrix
    int * mat_ptr = jm + line * size_d + *(pv + line);

    // initializing IVM
    if (state < 0) {
        if (*mat_ptr < 0) { // aka pruningCellState
            state = 1;
            for(int i=line+1;i<size_d;i++){
                pv[i] = 0;
            }
        }else{
            if (line < size_d - 2) {
                line++;
                generateLine2(jm, pv, dirVecs_d + ivm * size_d, line, state);
            } else {
                state = 1;
            }
        }
    }

    if (state > 0) {
        int l     = 0;          // first split [pos,end]
        while (pv[l] == ev[l] && l < size_d) l++;

        int * pos = pv + line; // current pos

        state = 0;
        while (pv[l] <= ev[l]) {              // approx check for (pos < end ?)
            //END OF LINE
            if (*pos >= (size_d - line)) {
                if (line == 0) break;      // cannot go up -> interval empty
                *pos = 0;
                line--;                                // aka goUp
                pos--;                                    // update current pos
                mat_ptr = jm + line * size_d + (*pos); // update pos in matrix (backtrack)
                *mat_ptr = negative_d(*mat_ptr);
            }
            else if (*mat_ptr < 0) // aka pruningCellState
            {
                assert(pv[line] < size_d);
                (*pos)++;  // aka goRight
                mat_ptr++; // update pos in matrix (next right)
            } else {
                assert(jm[line * size_d + pv[line]] >= 0);
                assert(line < size_d - 1);

                // found a node to bound --- check validity and set flag to "not empty"
                if (beforeEndPart(pv, ev, l)) {
                    atomicInc(&countNodes_d, INT_MAX);//atomic global counter
                    count[ivm]++;//per IVM counter
                    state = 1;
                }
                break;
            }
        }

        if (state == 1) {
            assert(line < size_d - 1);
            // seqGoDown(jm, pv, &dirVecs_d[ivm*size_d], line);
            line++;
            generateLine2(jm, pv, dirVecs_d + ivm * size_d, line, state);
        }
    }

    // increment statistics counters
    if (state > 0) atomicInc(&counter_d[exploringState], INT_MAX);
    if (state == 0) atomicInc(&counter_d[emptyState], INT_MAX);
    if (state < 0) atomicInc(&counter_d[initState], nbIVM_d);

    state_d[ivm] = state;
    line_d[ivm]  = line;
}


/*
 * 4(N+2)*sizeof(int) shared memory
 *
 * countNodes_d : count decomposed nodes
 * counter_d : count current states
 */
template<unsigned NN>
__global__ void
goToNext2(int * jobMats_d, int * posVecs_d, int * endVecs_d, int * dirVecs_d, int * line_d, int * state_d, unsigned long long int * count, unsigned int * counter_d)
{
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / g.size(); // global ivm id
    int thPos = g.thread_rank();
    int warpID = threadIdx.x / g.size();

    //shared memory...... pos,end,state,line
    //======================================================
    extern __shared__ int shar[];
    int * pv = shar;
    int * ev  = (int *) &pv[NN * size_d];
    int * state = (int *) &ev[NN * size_d];
    int * line  = (int *) &state[NN];

    // int * pv   = posv + warpID * size_d;            // shared mem - vector
    pv += warpID * size_d;
    ev += warpID * size_d;

    //load to shared memory ...
    for (int i = g.thread_rank(); i < size_d; i+=warpSize){
        pv[i] = posVecs_d[ivm * size_d + i];
        // posv[warpID * size_d + i] = posVecs_d[ivm * size_d + i];
        ev[i] = endVecs_d[ivm * size_d + i];
    }
    if (thPos == 0) {
        state[warpID] = state_d[ivm];
        line[warpID]  = line_d[ivm];
    }

    // pointers to IVM (just to compute less indices)
    g.sync();
    int * jm   = jobMats_d + ivm * size_d * size_d; // global mem matrix
    //======================================================

    // initializing IVM
    //======================================================
    if (state[warpID] < 0){
        int ret = initStep(g,jm,pv,dirVecs_d + ivm * size_d,line[warpID],state[warpID]);
        if (g.thread_rank() == 0){
            count[ivm]+=ret;
        }
    }
    g.sync();
    //======================================================

    // exploring IVM
    //======================================================
    if (state[warpID] > 0) {
        exploreStep(g,jm,pv,ev,line[warpID],state[warpID]);
        g.sync();
        if (state[warpID] == 1) {
            for (int i = g.thread_rank(); i< size_d; i += g.size()){
                assert(line[warpID] < size_d - 1);
                parallelGoDown(jm, pv, line[warpID], i);
                pv[line[warpID]]=0;
            }
            g.sync();
            if (g.thread_rank() == 0){
                count[ivm]+=1;//per IVM counter
                atomicInc(&countNodes_d, INT_MAX);//atomic global counter
                line[warpID]++;
            }
        }
    }
    g.sync();

    // increment statistics counters
    if (g.thread_rank() == 0) {
        if (state[warpID] > 0) atomicInc(&counter_d[exploringState], INT_MAX);
        if (state[warpID] == 0) atomicInc(&counter_d[emptyState], INT_MAX);
        if (state[warpID] < 0) atomicInc(&counter_d[initState], nbIVM_d);
    }

    g.sync();
//    __syncthreads();

    // back to global mem
    for (int i = thPos; i < size_d; i+=g.size()) {
        posVecs_d[ivm * size_d + i] = pv[i];
    }
    state_d[ivm] = state[warpID];
    line_d[ivm]  = line[warpID];
}
