__global__ void computeLength(const int *posVecs, const int *endVecs, int *length, const int *state, int *sumLength_d) {
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());

    int ivm   = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // global ivm id
    int thPos = g.thread_rank();                            // threadId within IVM
    int i = 0;

    if (ivm < nbIVM_d){
        for (i = thPos; i < size_d; i+=g.size()) {
            length[index2D(i, ivm)] = 0;
        }
        // compute lengths of exploring intervals / single thread
        if (thPos == 0 && state[ivm] == 1) {
            computeLen(length+ivm*size_d, posVecs+ivm*size_d, endVecs+ivm*size_d);
        }
        // warp parallel add
        for (i = thPos; i < size_d; i+=warpSize) {
            if(state[ivm] == 1)
                atomicAdd(&sumLength_d[i], (int)length[index2D(i, ivm)]);
        }

        // reset counter
        gpuBalancedIntern = 0;
    }
}
// ================================================
__global__ void computeMeanLength(int *sumLength, int *meanLength,
                                  float search_cut,
                                  int nonempty) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // completely sequential
    if (tid == 0) {
        int r = 0;

        // adjust sumLength to a correct factoradic number
        for (int j = size_d - 1; j > 0; j--) {
            if (sumLength[j] > (size_d - 1 - j)) {
                sumLength[j - 1] += sumLength[j] / (size_d - j);
                sumLength[j]     %= (size_d - j);
            }
        }

        // divide sumLength by #nonempty-IVM
        for (int i = 0; i < size_d; i++) {
            meanLength[i] =
                (int)(search_cut * (sumLength[i] + r * (size_d - i))) / nonempty;
            r = (int)(search_cut * (sumLength[i] + r * (size_d - i))) % nonempty;
        }

        // if (search_cut == 0)
        meanLength[size_d - 2] += 1;

    for (int i = 0; i < size_d; i++)
      sumLength[i] = 0;
    }
}

// --------------------------------------------------------
template < typename T >
__global__ void prepareShare(const T *state_d, int *flag, int *victim, const int *length_d, const int *meanLength_d, int off, int s, int r) {
    register int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t = (1 << s);

    //  int r=s*dim;
  if(state_d[tid] == 0 && tid<nbIVM_d){
    register int prec = (((tid>>r)&(t-1))>=off)?(tid-off*(1<<r)):(tid-off*(1<<r)+(t<<r));
    if (state_d[prec] != 0 && flag[prec] == 0){
            if (!isSmaller(length_d + prec * size_d, meanLength_d)) {
                flag[prec]  = 1;
                flag[tid]   = 1;
                victim[tid] = prec;
            }
        }
    }
}

// GPU work transfer without init
template <typename T>
__global__ void share_on_gpu2(T *jobMat, T *pos, T *end, T *dir, T *line, int numerator, int denominator, T *state, int *victim_flag, int *victim, unsigned int *ctrl_d)
{
    int thPos = threadIdx.x % warpSize;
    int ivm = blockIdx.x * PERBLOCK + threadIdx.x / warpSize;

    int vict = victim[ivm];
    int i    = 0;
    int l    = 0;

    if (state[ivm] == 0 && state[vict] == 1) {
        while(pos[index2D(l,vict)] == end[index2D(l,vict)] && l < line[vict] && l < size_d - 6) l++;

        if (pos[index2D(l, vict)] >= end[index2D(l, vict)]) {
            state[ivm] = 0;
        }
        else {
            for (int k = 0; k <= (size_d / warpSize); k++) {
                if (k * warpSize + thPos < l) {
                    pos[index2D(k*warpSize+thPos, ivm)] = pos[index2D(k*warpSize+thPos, vict)];
                    dir[index2D(k*warpSize+thPos, ivm)] = dir[index2D(k*warpSize+thPos, vict)];
                }
                for (i = 0; i < l; i++) {
                    if(k*warpSize + thPos < size_d)jobMat[index3D(i, k*warpSize + thPos, ivm)] = jobMat[index3D(i, k*warpSize + thPos, vict)];
                }
                if (k * warpSize + thPos < size_d) {
                    end[index2D(k * warpSize + thPos, ivm)] = end[index2D(k * warpSize + thPos, vict)];
                    jobMat[index3D(l, k * warpSize + thPos, ivm)] = jobMat[index3D(l, k * warpSize + thPos, vict)];
                }
            }
            //}

            if (thPos == 0) {
                dir[index2D(l, ivm)] = dir[index2D(l, vict)];
                pos[index2D(l, ivm)] = cuttingPosition(l, denominator, pos + vict * size_d, end + vict * size_d, jobMat + vict * size_d * size_d);
                end[index2D(l, vict)] = pos[index2D(l, ivm)] - 1;

                for (i = l + 1; i < size_d; i++) {
                    pos[index2D(i, ivm)]  = 0;
                    end[index2D(i, vict)] = size_d - i - 1;
                }
                line[ivm]  = l;
                state[ivm] = 1;
                atomicInc(&gpuBalancedIntern, INT_MAX);

                //        atomicInc(&ctrl_d[7], INT_MAX);
            }
        }
    }

    // reset victim
    victim[ivm] = ivm;

    // reset counter
    victim_flag[ivm] = 0;
} // share_on_gpu2
