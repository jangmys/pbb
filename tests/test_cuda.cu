#include <cuda.h>
#include <cooperative_groups.h>

#include "../gpu/include/reductions.cuh"

using namespace cooperative_groups;


__global__ void cuda_sum(int* result, int* arr, int N)
{
    //thread indexing
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    //tile-level reduction : lane 0 gets tile-sum
    int val = tile_sum<32>(g, arr[tid]);

    if(g.thread_rank()==0){
        atomicAdd(result,val);
    }
}

__global__ void cuda_sum2(int* result, int* arr, int N)
{
    //thread indexing
    thread_block bl = this_thread_block();
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    thread_block_tile<32> g = tiled_partition<32>(bl);
    int lane = g.thread_rank();
    int warpID = g.meta_group_rank();

    //shared memory (per block)
    __shared__ int sum_smem[32];
    if(tid<32){ sum_smem[tid] = 0; }
    __syncthreads();

    if(tid >= N)return;

    //tile-level reduction : lane 0 gets tile-sum
    int val = tile_sum<32>(g, arr[tid]);
    if(lane == 0){
        sum_smem[warpID] = val;
    }

    //sync block
    __syncthreads();

    //first warp reduces tile-sums across block
    //lane 0 in warp 0 atomic add
    if(warpID == 0){
        val = tile_sum<32>(g, sum_smem[bl.thread_rank()]);
        g.sync();
        if(lane==0){
            atomicAdd(result,val);
        }
    }
}



int main(int argc, char **argv)
{
    int N = 1 << 30;
    // int N = 1 << 30;

    struct timespec t1,t2;

    int *sum;
    int *arr;

    cudaMallocManaged(&sum,sizeof(int));
    cudaMallocManaged(&arr,N*sizeof(int));
    for(unsigned i=0;i<N;i++){
        arr[i] = -N/2 + 1 + i;
    }

    clock_gettime(CLOCK_MONOTONIC,&t1);
    for(unsigned i=0;i<N;i++){
        *sum += arr[i];
    }
    clock_gettime(CLOCK_MONOTONIC,&t2);
    printf("%2.9f ms\n",1000*(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e6);
    printf("%d\n",*sum);

    *sum = 0;

    int blocksize = 128;
    if(argc>1)
        blocksize = atoi(argv[1]);

    int nblocks = (N+blocksize-1)/blocksize;
    // int sharedbytes = blocksize * sizeof(int);

    //FIRST kernel launch ... force memcpy
    cuda_sum<<<nblocks,blocksize>>>(sum,arr,2);
    cudaDeviceSynchronize();
    *sum = 0;

    clock_gettime(CLOCK_MONOTONIC,&t1);
    cuda_sum<<<nblocks,blocksize>>>(sum,arr,N);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC,&t2);
    printf("%2.9f ms\n",1000*(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e6);
    printf("%d\n",*sum);

    *sum = 0;

    clock_gettime(CLOCK_MONOTONIC,&t1);
    cuda_sum2<<<nblocks,blocksize>>>(sum,arr,N);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC,&t2);
    printf("%2.9f ms\n",1000*(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e6);
    printf("%d\n",*sum);
}
