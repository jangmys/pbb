#include "gpu_helper.cuh"

#include <cooperative_groups.h>

#define FULL_MASK 0xffffffff

using namespace cooperative_groups;


template <unsigned size, typename T>
__device__ T tile_sum(thread_block_tile<size> g, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = size / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }
    g.sync();

    return val; // note: only thread 0 will return full sum
}

template <unsigned size>
__device__ int tile_min(thread_block_tile<size> g, int val)
{
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = size / 2; i > 0; i /= 2) {
        val = min(val, g.shfl_down(val, i));
    }
    g.sync();

    return val; // note: only thread 0 will return min
}

// sort arr[ [lim1,lim2 ] ]
template <unsigned size>
inline __device__ void
tile_EvenOddSortIncr(thread_block_tile<size> g,int * arr, const int * key, const int nbelem)
{
    int lane = g.thread_rank();

    bool sorted = false;

    while(!sorted)
    {
        sorted=true;

        //even
        for (int i = 2*lane; i<nbelem-1; i += g.size())
        {
            if(key[arr[i]] > key[arr[i+1]]){
                swap_d(&arr[i],&arr[i+1]);
                sorted=false;
            }
        }
        g.sync();

        //odd
        for (int i = 2*lane+1; i < nbelem-1; i += g.size()) {
            if(key[arr[i]] > key[arr[i+1]]){
                swap_d(&arr[i],&arr[i+1]);
                sorted=false;
            }
        }
        g.sync();

        sorted = g.all(sorted);
        g.sync();
    }
}

template <unsigned size>
inline __device__ void
tile_EvenOddSortDecr(thread_block_tile<size> g,int * arr, const int * key, const int nbelem)
{
    int lane = g.thread_rank();

    bool sorted = false;

    while(!sorted)
    {
        sorted=true;

        //even
        for (int i = 2*lane; i<nbelem-1; i += g.size())
        {
            if(key[arr[i]] < key[arr[i+1]]){
                swap_d(&arr[i],&arr[i+1]);
                sorted=false;
            }
        }
        g.sync();

        //odd
        for (int i = 2*lane+1; i < nbelem-1; i += g.size()) {
            if(key[arr[i]] < key[arr[i+1]]){
                swap_d(&arr[i],&arr[i+1]);
                sorted=false;
            }
        }
        g.sync();

        sorted = g.all(sorted);
        g.sync();
    }
}

template < typename T >
__global__ void checkEnd(T *state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int warpID = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;

    int end = state[tid];

    //====================
    static __shared__ int endblock[32];
    if(lane==0)
        endblock[warpID]=0;//init
    __syncthreads();

    end = __any_sync(FULL_MASK,end);        // warp reduce
    if (lane == 0) endblock[warpID] = end;  // Write reduced value to shared memory
    __syncthreads();                        // wait for partial reductions

    end = (threadIdx.x < blockDim.x / warpSize) ? endblock[lane] : 0;
    if (warpID == 0) end = __any_sync(FULL_MASK,end);  // Final reduce within first warp
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMax(&deviceEnd, end);
    }
}

__global__ void reduce(const int *todo_d, int *tempArr, int *auxArr, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int pout = 0, pin = 1;

    extern __shared__ int shArr[];

    // if(bid*n+tid >= nbIVM_d)return;

    shArr[pout * n + tid] = (bid * n + tid > 0) ? todo_d[bid * n + tid - 1] : 0; //compute prefix sum
    __syncthreads();

    // reduction on shared mem arrays
    for (int off = 1; off < n; off *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pout;

        if (tid >= off) {
            shArr[pout * n + tid] = shArr[pin * n + tid - off] + shArr[pin * n + tid];
        } else {
            shArr[pout * n + tid] = shArr[pin * n + tid];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        auxArr[bid] = shArr[pout * n + n - 1]; // last value of block
    }

    // reduced todo back to global
    tempArr[bid * n + tid] = shArr[pout * n + tid];
}

__global__ void reduce2(int *tempArr, const int *auxArr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < blockIdx.x; i++) {
        tempArr[tid] += auxArr[i];
    }
}
