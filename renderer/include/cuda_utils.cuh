#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#define CUDA_CALL(x) x
#define CUDA_CALL(x) x
#define ASSERT(x) void(0)

#if USE_DOUBLE_PRECISION
typedef double3 Float3;
#else
typedef float3 Float3;
#endif

#endif /* CUDA_UTILS_H_ */
