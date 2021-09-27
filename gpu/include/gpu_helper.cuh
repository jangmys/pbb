#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <unistd.h>
#include <stdio.h>

#include <cooperative_groups.h>
using namespace cooperative_groups;
namespace cg = cooperative_groups;


__device__ unsigned int todo;
__device__ unsigned int deviceEnd;
__device__ unsigned int targetNode;
__device__ unsigned int gpuBalancedIntern;
__device__ unsigned int countNodes_d;

__constant__ int _boundMode;
__constant__ int _nbJob;
__constant__ int nbIVM_d;
__constant__ int size_d;

__device__ enum counterID { exploringState, emptyState, initState };
__device__ enum ctrlID { toDo, gpuEnd, foundLeaf, bestCost };

// ____________________________________________________________
template <typename T>
inline __device__ T
negative_d(const T i)
{
    if (i < 0)
        return i;
    else
        return (-i) - 1;
}

// ___________________________________________________________
template <typename T>
inline __device__ T
absolute_d(const T i)
{
    if (i >= 0)
        return i;
    else
        return (-i) - 1;
}

// __________________________________________________________
template <typename T>
inline __device__ void
swap_d(T * ptrA, T * ptrB)
{
    T tmp = *ptrA;

    *ptrA = *ptrB;
    *ptrB = tmp;
}

// =======================================
// wrapper: a[ivm][row][col] --> a[index3D(row,col,ivm)]
inline __device__ int
index3D(const int row, const int col, const int ivm)
{
    return (ivm * size_d * size_d + row * size_d + col);
}

// =======================================
// wrapper: a[ivm][row] --> a[index2D(ivm,row)]
inline __device__ int
index2D(const int row, const int ivm)
{
    return (ivm * size_d + row);
}

// =======================================
// returns absolute value of (a)
template <typename T>
inline __device__ T
abso(const T a){ return (a < 0) ? (-a) : a; }

// =======================================
// returns maximum of two int values
inline __device__ int
maxi(const int a, const int b){ return (a < b) ? b : a; }

// =======================================
// returns minimum of two int values
inline __device__ int
mini(const int a, const int b){ return (a > b) ? b : a; }

__forceinline__ __device__ unsigned int
laneID()
{
    unsigned int lane;

    asm volatile ("mov.u32 %0, %%laneid;" : "=r" (lane));
    return lane;
}

__device__ void
gnomeSortSequential(int * arr, int * key, int lim1, int lim2)
{
    int i = lim1;
    int j = lim1 + 1;

    while (i < lim2) {
        if (key[arr[i - 1]] > key[arr[i]]) {
            swap_d(&arr[i - 1], &arr[i]);
            if (--i) continue;
        }
        i = j++;
    }
}

int
get_available_devices(int& nb_block, int& block_size)
{
    cudaDeviceProp prop;
    // //	std::cout<<"aabbbbbbbbbbbaaaa"<<std::endl<<std::flush;
    int count;

    // //	new bound_flowshop();
    cudaGetDeviceCount(&count);
    //  posVect[ind2D(0, 0)] = size;
    printf("\n\n   --- Number of device %d ---\n\n", count);
    //
    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("   --- General Information for device %d ---\n\n", i);
        printf("Name:  %s\n", prop.name);
        printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
        printf("Clock rate:  %d\n", prop.clockRate);
        printf("Total global mem:  %lu\n", prop.totalGlobalMem);
        printf("Total constant Mem:  %lu\n", prop.totalConstMem);
        printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp:  %lu\n", prop.sharedMemPerBlock);
        printf("Registers per mp:  %d\n", prop.regsPerBlock);
        printf("Threads in warp:  %d\n", prop.warpSize);
        printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
          prop.maxThreadsDim[2]);
        printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }

    nb_block   = prop.maxThreadsDim[0] / 2;
    block_size = prop.maxThreadsPerBlock / 2;
    // freemem();
    return count;
} // get_available_devices
