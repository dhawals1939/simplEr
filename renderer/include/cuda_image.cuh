#ifndef CUDA_IMAGE_H_
#define CUDA_IMAGE_H_

#include "constants.h"
#include "image.h"

#include "cuda_utils.cuh"

namespace cuda {

/* 2D Image class. Not for use as a rendering target. */
template <typename T>
class Image2 {
public:

    /* Create a cuda::Image2 on the GPU. */
    __host__ static Image2* from(const image::Image2<T> &image) {
        Image2 cuda_image = new Image2(image.getXRes(), image.getYRes());
        cuda_image.setPixels(image.getImage(), image.getXRes(), image.getYRes());

        Image2 *d_cuda_image;
        CUDA_CALL(cudaMalloc((void **)&d_cuda_image, sizeof(Image2)));
        CUDA_CALL(cudaMemcpy(d_cuda_image, &cuda_image, sizeof(CudaClass), cudaMemcpyHostToDevice));
        return d_cuda_image;
    }

	__host__ __device__ inline int getXRes() const {
		return m_xRes;
	}

	__host__ __device__ inline int getYRes() const {
		return m_yRes;
	}

    __device__ inline void ind2sub(const int &ndx, int &x, int &y) const {
		ASSERT(ndx >= 0 && ndx < m_xRes * m_yRes);
		y = ndx/m_xRes;
		x = ndx - y*m_xRes;
	}

	__host__ ~Image2() {
        if (m_pixels)
            CUDA_CALL(cudaFree(m_pixels));
	}

private:
    __host__ Image2(int xRes, int yRes) : m_xRes(xRes), m_yRes(yRes) {
        CUDA_CALL(cudaMalloc((void **)&m_pixels,
                             m_xRes * m_yRes * sizeof(T)));
    }

	__host__ inline void setPixels(const T *host_buffer, size_t size) {
        ASSERT(size == m_xRes * m_yRes);
        CUDA_CALL(cudaMemcpy(m_pixels, host_buffer,
                             m_xRes * m_yRes * sizeof(T),
                             cudaMemcpyHostToDevice));
	}

    int m_xRes;
    int m_yRes;
    T *m_pixels;
};

}


#endif /* CUDA_IMAGE_H_ */
