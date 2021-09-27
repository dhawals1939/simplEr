/*
 * cuda_renderer.cu
 *
 *  Created on: Aug 2, 2021
 *      Author: Andre
 */

#include "cuda_renderer.h"
#include "cuda_vector.cuh"
#include "cuda_utils.cuh"

namespace cuda {

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
    int num_threads = blockDim.x;

    // TODO: Zero out constants.image

    TVector3<Float> pos;
    TVector3<Float> dir;
    Float totalDistance;
    short uses = 0;

    Float weight = d_constants.weight;
    Scene *scene = d_constants.scene;

    if (scene.genRay(pos, dir, totalDistance, uses)) {
		  float scaling = max(min(sinf(scene.getUSPhi_min() + scene.getUSPhi_range()*scene.sampler(uses)), scene.getUSMaxScaling()), -scene.getUSMaxScaling());
          (void)scaling;
    //    Assert(!m_useDirect);
    //    if(m_useDirect)
    //        directTracing(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, totalDistance); // Traces and adds direct energy, which is equal to weight * exp( -u_t * path_length);
    //    scatter(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, totalDistance, *costFunctions[id], problem[id], initializations+id*3);
    }

}

void CudaRenderer::renderImage(image::SmallImage& target, const med::Medium &medium, const scn::Scene &scene, int numPhotons) {
    setup(target, medium, scene, numPhotons);

    renderPhotons<<<1,numPhotons>>>();
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(image, h_constants.cudaImage,
                         target.getXRes()*target.getYRes()*target.getZRes()*sizeof(float),
                         cudaMemcpyDeviceToHost));

    target.copyImage(image, target.getXRes()*target.getYRes()*target.getZRes());

    cleanup();
}

/* Allocates host and device data and sets up RNG. */
//TODO: introduce medium
void CudaRenderer::setup(image::SmallImage& target, const med::Medium &medium, const scn::Scene &scene, int numPhotons) {
    /* Allocate host memory */
    image = new float[target.getXRes()*target.getYRes()*target.getZRes()*sizeof(float)];

    /* Allocate device memory*/
    float* cudaImage, cudaRandom;
    CUDA_CALL(cudaMalloc((void **)&cudaImage,
                         target.getXRes()*target.getYRes()*target.getZRes()*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&cudaRandom, requiredRandomNumbers(numPhotons) * sizeof(float)));

    /* Setup generator. */
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937));

    /* Send in parameters to device */
    h_constants.image = cudaImage;
    h_constants.random = cudaRandom;
    h_constants.scene = Scene::from(scene);
    h_constants.medium = Medium::from(medium);
    h_constants.weight = photon::getWeight(medium, scene, numPhotons);

    CUDA_CALL(cudaMemcpyToSymbol(d_constants, &h_constants, sizeof(Constants)));

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
    if (image) delete[] image;

    if (generator) CURAND_CALL(curandDestroyGenerator(generator));

    Constants::free(h_constants);
}

CudaRenderer::~CudaRenderer() {}

/* Required amount of random numbers to run the renderPhotons kernel on numPhotons */
unsigned int CudaRenderer::requiredRandomNumbers(int numPhotons) {
    return numPhotons * RANDOM_NUMBERS_PER_PHOTON;
}

}
