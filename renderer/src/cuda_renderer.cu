/*
 * cuda_renderer.cu
 *
 *  Created on: Aug 2, 2021
 *      Author: Andre
 */

#include "cuda_renderer.h"

#include <cuda.h>
#include <cuda_runtime.h>


/* TODO:
 * - The contribution of a photon equates to a sum on the pixel(s) it affects. Figure
 * out how to synchronize the contributions across all photons. */

// Rendering a photon requires 6 calls to sampler() or
// equivalently, 6 random floats/doubles
#define RANDOM_NUMBERS_PER_PHOTON 6

namespace cuda {

// Store symbols here so as to avoid long argument list
// in kernel calls
struct Constants {
    float* image;
    float* random; // TODO: Add const qualifier to random?
};

__constant__ Constants constants;

__global__ void renderPhotons() {
    //int i = threadIdx.x;

}

void CudaRenderer::renderImage(image::SmallImage& target, int numPhotons) {
    setup(target, numPhotons);

    renderPhotons<<<1,numPhotons>>>();
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(image, cudaImage,
                         target.getXRes()*target.getYRes()*target.getZRes()*sizeof(float),
                         cudaMemcpyDeviceToHost));

    // TODO: Write image to target

    // TODO: Implement this functionallity withing kernel
    //if (scene.genRay(pos, dir, sampler[id], totalDistance)) {

	//	float scaling = std::max(std::min(std::sin(scene.getUSPhi_min() + scene.getUSPhi_range()*sampler[id]()), scene.getUSMaxScaling()), -scene.getUSMaxScaling());

    //    Assert(!m_useDirect);
    //    if(m_useDirect)
    //        directTracing(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, totalDistance); // Traces and adds direct energy, which is equal to weight * exp( -u_t * path_length);
    //    scatter(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, totalDistance, *costFunctions[id], problem[id], initializations+id*3);
    //}

	//img.mergeImages(img0);

    //cleanup();
}

/* Allocates host and device data and sets up RNG. */
void CudaRenderer::setup(image::SmallImage& target, int numPhotons) {
    /* Allocate host memory */
    image = new float[target.getXRes()*target.getYRes()*target.getZRes()*sizeof(float)];

    /* Allocate device memory*/
    CUDA_CALL(cudaMalloc((void **)&cudaImage,
                         target.getXRes()*target.getYRes()*target.getZRes()*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&cudaRandom, requiredRandomNumbers(numPhotons) * sizeof(float)));

    /* Setup generator. */
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937));

    /* Send in parameters to device */
    Constants params;
    params.image = cudaImage;
    params.random = cudaRandom;

    CUDA_CALL(cudaMemcpyToSymbol(constants, &params, sizeof(Constants)));

    /* Generate random numbers to be used by each thread */
    genDeviceRandomNumbers(requiredRandomNumbers(numPhotons));
}

/* Generates random numbers on the device. */
// TODO: currently sequential
void CudaRenderer::genDeviceRandomNumbers(int num, CudaSeedType seed) {
    smp::SamplerSet sampler(1, 0);
    float *random = new float[num];
    for (int i = 0; i < num; i++) {
        random[i] = sampler[0]();
    }

    CUDA_CALL(cudaMemcpy(cudaRandom, random, sizeof(float)*num, cudaMemcpyHostToDevice));

    delete[] random;

    // TODO: Enable below to make it parallel
    //CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));
    ///* Generate reals uniformly between 0.0 and 1.0 */
    //CURAND_CALL(curandGenerateUniform(generator, cudaRandom, num));
}

void CudaRenderer::cleanup() {
    if (image) {
        delete[] image;
    }

    if (cudaImage) {
        CURAND_CALL(curandDestroyGenerator(generator));

        CUDA_CALL(cudaFree(cudaImage));
        CUDA_CALL(cudaFree(cudaRandom));
    }
}

CudaRenderer::~CudaRenderer() {}

/* Required amount of random numbers to run the renderPhotons kernel on numPhotons */
unsigned int CudaRenderer::requiredRandomNumbers(int numPhotons) {
    return numPhotons * RANDOM_NUMBERS_PER_PHOTON;
}

}
