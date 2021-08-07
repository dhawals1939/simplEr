/*
 * cudaRenderer.cu
 *
 *  Created on: Aug 2, 2021
 *      Author: Andre
 */

#include "cudaRenderer.h"

/* TODO:
 * - The contribution of a photon equates to a sum on the pixel(s) it affects. Figure
 * out how to synchronize the contributions across all photons.
 * - Random number generation has to match that of the boost library. In the future,
 * support for the sse rng as well. */

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

// Store symbols here so as to avoid long argument list
// in kernel calls
struct Constants {
    float* image;
    float* random;
};

__constant__ Constants constants;

__global__ void renderPhotons() {
    //int i = threadIdx.x;

}

void CudaRenderer::renderImage(image::SmallImage& img, int numPhotons) {

    setup(img);

    renderPhotons<<<1,numPhotons>>>();
    CUDA_CALL(cudaDeviceSynchronize());

}

/* Allocates device data, sends parameters to device and sets up RNG. */
void CudaRenderer::setup(image::SmallImage& img, int numPhotons) {
    CUDA_CALL(cudaMalloc((void **)&cudaImage,
                         img.getXRes()*img.getYRes()*img.getZRes()*sizeof(float)));

    genDeviceRandomNumbers();

    /* Send in parameters to device */
    Constants params;
    params.image = cudaImage;
    params.random = cudaRandom;

    CUDA_CALL(cudaMemcpyToSymbol(constants, &params, sizeof(Constants)));

}

// TODO: Ensure ordering and offset are the same as boost
/* Generates random numbers on the device. */
void CudaRenderer::genDeviceRandomNumbers(CudaSeedType seed = CudaSeedType(5489)) {
    int num = requiredRandomNumbers(numPhotons);
    CUDA_CALL(cudaMalloc((void **)&cudaRandom, num * sizeof(float)));

    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));

    /* Generate reals uniformly between 0.0 and 1.0 */
    CURAND_CALL(curandGenerateUniform(generator, cudaRandom, num));
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (cudaImage) {
        CURAND_CALL(curandDestroyGenerator(generator));

        CUDA_CALL(cudaFree(cudaImage));
        CUDA_CALL(cudaFree(cudaRandom));
    }

    // No need to free constant memory
}

/* Required amount of random numbers to run the renderPhotons kernel on numPhotons */
unsigned int requiredRandomNumbers(unsigned int numPhotons) {
    return numPhotons * 4;
}
