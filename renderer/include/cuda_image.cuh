#ifndef CUDA_IMAGE_H_
#define CUDA_IMAGE_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_vector.cuh"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    }} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    }} while(0)

namespace cuda {

/* 2D Image class. Not for use as a rendering target. */
template <typename T>
class Image {
public:

    __host__ Image(int xRes, int yRes) : m_xRes(xRes), m_yRes(yRes) {
        CUDA_CALL(cudaMalloc((void **)&m_pixels,
                             m_xRes * m_yRes * sizeof(float)));
    }

	__host__ inline void setPixels(T *_buffer) {
        CUDA_CALL(cudaMemcpy(m_pixels, host_buffer,
                             m_xRes * m_yRes * sizeof(Float),
                             cudaMemcpyHostToDevice));
	}

	__device__ inline void ind2sub(const int &ndx, int &x, int &y) const {
		Assert(ndx >= 0 && ndx < m_xRes*m_yRes);
		y = ndx/m_xRes;
		x = ndx - y*m_xRes;
	}

private:
    int m_xRes;
    int m_yRes;
    T *m_pixels;
};

}


#endif /* CUDA_IMAGE_H_ */
