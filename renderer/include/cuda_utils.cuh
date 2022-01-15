#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d, %s\n",__FILE__,__LINE__, cudaGetErrorString(x));\
    }} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    }} while(0)

//#define ASSERT(x) do { if (!(x)) { \
//    printf("Error at %s:%d\n",__FILE__,__LINE__);\
//    }} while(0)

//#define CUDA_CALL(x) x
//#define CUDA_CALL(x) x
#define ASSERT(x) void(0)

#ifdef USE_DOUBLE_PRECISION
typedef double3 Float3;
#else
typedef float3 Float3;
#endif

#endif /* CUDA_UTILS_H_ */
