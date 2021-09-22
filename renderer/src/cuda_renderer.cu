/*
 * cuda_renderer.cu
 *
 *  Created on: Aug 2, 2021
 *      Author: Andre
 */

#include "cuda_vector.cuh"
#include "cuda_renderer.h"
#include "cuda_utils.cuh"

namespace cuda {

// Store symbols here so as to avoid long argument list
// in kernel calls
struct Constants {
    float* image;
    float* random; // TODO: Add const qualifier to random?
};

__constant__ Constants constants;

__device__ bool deflect(pos, dir, totalDistance) {
    ///* Deflection computation:
    // * Point going through the center of lens and parallel to dir is [pos.x, 0, 0]. Ray from this point goes straight
    // * This ray meets focal plane at (pos.x - d[0] * f/d[0], -d[1] * f/d[0], -d[2] * f/d[0]) (assuming -x direction of propagation of light)
    // * Original ray deflects to pass through this point
    // * The negative distance (HACK) travelled by this ray at the lens is -f/d[0] - norm(focalpoint_Pos - original_Pos)
    // */
    //Float squareDistFromLensOrigin = 0.0f;
    //for(int i = 1; i < pos.dim; i++)
    //	squareDistFromLensOrigin += pos[i]*pos[i];
    //if(squareDistFromLensOrigin > m_squareApertureRadius)
    //	return false;

    //Assert(pos.x == m_origin.x);
    //Float invd = -1/dir[0];
    //dir[0] = -m_focalLength;
    //dir[1] = dir[1]*invd*m_focalLength - pos[1];
    //dir[2] = dir[2]*invd*m_focalLength - pos[2];
    //totalDistance += m_focalLength*invd - dir.length();
    //dir.normalize();
    //return true; // should return additional path length added by the lens.
    return false;
}

// FIXME: Actually propagateTillLens
__device__ bool propagateTillMedium() {
    //Float dist = -(pos[0]-m_origin[0])/dir[0];            //FIXME: Assumes that the direction of propagation is in -x direction.
    //pos += dist*dir;
    //totalDistance += dist;
    //if(m_active)
    //	return deflect(pos, dir, totalDistance);
    //else
    //	return true;
    return false;
}


__device__ void scatter() {
//	Assert(scene.getMediumBlock().inside(p));
//
//	if ((medium.getAlbedo() > FPCONST(0.0)) && ((medium.getAlbedo() >= FPCONST(1.0)) || (sampler() < medium.getAlbedo()))) {
//		VectorType<Float> pos(p), dir(d);
//
//		Float dist = getMoveStep(medium, sampler);
//		if (!scene.movePhoton(pos, dir, dist, totalOpticalDistance, sampler, scaling)) {
//			return;
//		}
//
//#ifdef PRINT_DEBUGLOG
//		std::cout << "dist: " << dist << "\n";
//		std::cout << "pos: (" << pos.x << ", " << pos.y << ", " << pos.z << ") " << "\n";
//		std::cout << "dir: (" << dir.x << ", " << dir.y << ", " << dir.z << ") " << "\n";
//#endif
//		int depth = 1;
//		Float totalDist = dist;
//		while ((m_maxDepth < 0 || depth <= m_maxDepth) &&
//				(m_maxPathlength < 0 || totalDist <= m_maxPathlength)) {
//			if(m_useAngularSampling)
//                scene.addEnergyInParticle(img, pos, dir, totalOpticalDistance, depth, weight, medium, sampler, scaling);
//			else
//				scene.addEnergy(img, pos, dir, totalOpticalDistance, depth, weight, medium, sampler, scaling, costFunction, problem, initialization);
//			if (!scatterOnce(pos, dir, dist, scene, medium, totalOpticalDistance, sampler, scaling)){
//#ifdef PRINT_DEBUGLOG
//				std::cout << "sampler after failing scatter once:" << sampler() << std::endl;
//#endif
//				break;
//			}
//#ifdef PRINT_DEBUGLOG
//			std::cout << "sampler after succeeding scatter once:" << sampler() << std::endl;
//
//			std::cout << "dist: " << dist << "\n";
//			std::cout << "pos: (" << pos.x << ", " << pos.y << ", " << pos.z << ", " << "\n";
//			std::cout << "dir: (" << dir.x << ", " << dir.y << ", " << dir.z << ", " << "\n";
//#endif
//#if USE_SIMPLIFIED_TIMING
//			totalOpticalDistance += dist;
//#endif
//			++depth;
//		}
//	}
}

__global__ void renderPhotons() {
    int i = threadIdx.x;

    // For now just compute photon contribution to this pixel
    // Later worry about reduction across CUDA threads

    // TODO: Zero out constants.image

    TVector3<Float> pos;
    TVector3<Float> dir;
    Float totalDistance;
    // TODO: Implement this functionality within kernel
    if (scene.genRay(pos, dir, totalDistance)) {

	//	float scaling = std::max(std::min(std::sin(scene.getUSPhi_min() + scene.getUSPhi_range()*sampler[id]()), scene.getUSMaxScaling()), -scene.getUSMaxScaling());

    //    Assert(!m_useDirect);
    //    if(m_useDirect)
    //        directTracing(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, totalDistance); // Traces and adds direct energy, which is equal to weight * exp( -u_t * path_length);
    //    scatter(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, totalDistance, *costFunctions[id], problem[id], initializations+id*3);
    }

}

void CudaRenderer::renderImage(image::SmallImage& target, int numPhotons) {
    setup(target, numPhotons);

    renderPhotons<<<1,numPhotons>>>();
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(image, cudaImage,
                         target.getXRes()*target.getYRes()*target.getZRes()*sizeof(float),
                         cudaMemcpyDeviceToHost));

	//img.mergeImages(img0);
    //cleanup();

    // TODO: Write image to target

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
// TODO: currently sequential, compare to result produced by sequential renderer (as opposed to threaded)
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
