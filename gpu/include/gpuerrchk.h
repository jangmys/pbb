#ifndef GPUERRCHK_H_
#define GPUERRCHK_H_

#include <cuda.h>
#include <cuda_runtime.h>
//#include "/usr/local/cuda-7.5/include/cuda.h"

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#endif 	
