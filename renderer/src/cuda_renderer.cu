/*
 * cuda_renderer.cu
 *
 *  Created on: Aug 2, 2021
 *      Author: Andre
 */

#include "cuda_renderer.h"
#include "cuda_vector.cuh"
#include "cuda_utils.cuh"
#include "cuda_scene.cuh"

namespace cuda {

// Store symbols here so as to avoid long argument list
// in kernel calls
struct Constants {
    Float *image;
    Float *random;
    Scene *scene;
    Medium *medium;
    Float weight;

    // TODO: Add maxDepth, maxPathlength, useDirect, useAngularSampling
};

__constant__ Constants d_constants;

__device__ Float Sampler::sample(short &uses) const{
    ASSERT(uses < RANDOM_NUMBERS_PER_PHOTON);
    ASSERT(threadIdx.x * RANDOM_NUMBERS_PER_PHOTON + uses + 1 < m_size);
    return m_random[threadIdx.x * RANDOM_NUMBERS_PER_PHOTON + uses++];
}

__device__ inline Float safeSqrt(Float x) {
    return x > FPCONST(0.0) ? sqrtf(x) : FPCONST(0.0);
}

__device__ inline void reflect(const TVector3<Float> &a, const TVector3<Float> &n,
                               TVector3<Float> &b) {
    b = -FPCONST(2.0)*dot(a, n)*n + a;
}

__device__ inline bool refract(const TVector3<Float> &a, const TVector3<Float> &n,
                        Float eta, TVector3<Float> &b) {
    TVector3<Float> q = dot(a,n)*n;
    TVector3<Float> p = (a-q)/eta;

    if (p.length() > FPCONST(1.0)) {
        reflect(a, n, b);
        return false;
    } else {
        q.normalize();
        q *= safeSqrt(FPCONST(1.0) - p.lengthSquared());
        b = p + q;
        return true;
    }
}

__device__ inline Float fresnelDielectric(Float cosThetaI, Float cosThetaT, Float eta) {
	if (fabsf(eta - FPCONST(1.0)) < M_EPSILON * max(FPCONST(1.0), fabsf(eta))) {
		return FPCONST(0.0);
	} else {
		Float Rs = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
		Float Rp = (cosThetaT - eta * cosThetaI) / (cosThetaT + eta * cosThetaI);

		return FPCONST(0.5) * Rs * Rs + Rp * Rp;
	}
}

__device__ inline void SmoothDielectric::sample(const TVector3<Float> &in, const TVector3<Float> &n,
				Sampler &sampler, TVector3<Float> &out, short &samplerUses) const {
	if (fabsf(m_ior1 - m_ior2) < M_EPSILON) {
		// index matched
		out = in;
	} else {
		Float eta;
		if (dot(in, n) < -M_EPSILON) {
			// entering ior2 from ior1
			eta = m_ior2/m_ior1;
		}
		else {
			// entering ior1 from ior2
			eta = m_ior1/m_ior2;
		}

		TVector3<Float> outT;
		if (!refract(in, n, eta, outT)) {
			// TIR
			out = outT;
		} else {
			TVector3<Float> outR;
			reflect(in, n, outR);

			Float cosI = absDot(n, in), cosT = absDot(n, outT);
			Float fresnelR = fresnelDielectric(cosI, cosT, eta);

			// return either refracted or reflected direction based on the Fresnel term
			out = (sampler(samplerUses) < fresnelR ? outR : outT);
		}
	}
}

// Sample random ray
__device__ bool AreaTexturedSource::sampleRay(TVector3<Float> &pos, TVector3<Float> &dir,
                                              Float &totalDistance, Sampler& sampler, short &samplerUses) const{
    pos = *m_origin;

    // sample pixel position first
	int pixel = m_textureSampler->sample(sampler(samplerUses));
	int p[2];
	m_texture->ind2sub(pixel, p[0], p[1]);

	// Now find a random location on the pixel
	for (int iter = 1; iter < m_origin->dim; ++iter) {
		pos[iter] += - (*m_plane)[iter - 1] / FPCONST(2.0) +
            p[iter - 1] * (*m_pixelsize)[iter-1] + sampler(samplerUses) * (*m_pixelsize)[iter - 1];
	}

	dir = *m_dir;

	//FIXME: Hack: Works only for m_dir = [-1 0 0]
	Float z   = sampler(samplerUses)*(1-m_ct) + m_ct;
	Float zt  = sqrtf(FPCONST(1.0)-z*z);
    // FIXME: FPCONST(M_PI) might be generating complaints here
	Float phi = sampler(samplerUses)*2*M_PI;
    // FIXME: operator[] overload here might be causing issues
	dir[0] = -z;
	dir[1] = zt*cosf(phi);
	dir[2] = zt*sinf(phi);

	return propagateTillMedium(pos, dir, totalDistance);
}


__device__ inline Float getMoveStep(const Medium *medium, short &uses) {
    Sampler &sampler = *d_constants.scene->sampler;
    return -medium->getMfp() * logf(sampler(uses));
}

__device__ void Scene::er_step(TVector3<Float> &p, TVector3<Float> &d, Float stepSize, Float scaling) const{
#ifndef OMEGA_TRACKING
    d += HALF * stepSize * dV(p, d, scaling);
    p +=        stepSize * d/m_us->RIF(p, scaling);
    d += HALF * stepSize * dV(p, d, scaling);
#else
    Float two = 2; // To avoid type conversion

    TVector3<Float> K1P = stepSize * dP(d);
    TVector3<Float> K1O = stepSize * dOmega(p, d);

    TVector3<Float> K2P = stepSize * dP(d + HALF*K1O);
    TVector3<Float> K2O = stepSize * dOmega(p + HALF*K1P, d + HALF*K1O);

    TVector3<Float> K3P = stepSize * dP(d + HALF*K2O);
    TVector3<Float> K3O = stepSize * dOmega(p + HALF*K2P, d + HALF*K2O);

    TVector3<Float> K4P = stepSize * dP(d + K3O);
    TVector3<Float> K4O = stepSize * dOmega(p + K3P, d + K3O);

    p = p + ONE_SIXTH * (K1P + two*K2P + two*K3P + K4P);
    d = d + ONE_SIXTH * (K1O + two*K2O + two*K3O + K4O);
#endif
}

__device__ void Scene::traceTillBlock(TVector3<Float> &p, TVector3<Float> &d, Float dist, Float &disx, Float &disy, Float &totalOpticalDistance, Float scaling) const{
	TVector3<Float> oldp, oldd;

    Float distance = 0;
    long int maxsteps = dist/m_us->er_stepsize + 1, i, precision = m_us->getPrecision();

    Float current_stepsize = m_us->er_stepsize;

    for(i = 0; i < maxsteps; i++){
    	oldp = p;
    	oldd = d;

    	er_step(p, d, current_stepsize, scaling);

    	// check if we are at the intersection or crossing the sampled dist, then, estimate the distance and keep going more accurately towards the boundary or sampled dist
    	if(!m_block->inside(p) || (distance + current_stepsize) > dist){
    		precision--;
    		if(precision < 0)
    			break;
    		p = oldp;
    		d = oldd;
    		current_stepsize = current_stepsize / 10;
    		i  = 0;
    		maxsteps = 11;
    	}else{
    		distance += current_stepsize;
#if !USE_SIMPLIFIED_TIMING
    		totalOpticalDistance += current_stepsize * m_us->RIF(p, scaling);
#endif
    	}
    }

    ASSERT(i < maxsteps);
    disx = 0;
    disy = distance;
}

__device__ void Scene::addEnergyInParticle(const TVector3<Float> &p, const TVector3<Float> &d, Float distTravelled,
                                           int &depth, Float val, Sampler &sampler, const Float &scaling, short &uses) const {

	TVector3<Float> p1 = p;

	TVector3<Float> dirToSensor;

//	if( (p.x-m_camera.getOrigin().x) < 1e-4) // Hack to get rid of inf problems for direct connection
//		return;
//
//	sampleRandomDirection(dirToSensor, sampler); // Samples by assuming that the sensor is in +x direction.
//
////	if(m_camera.getOrigin().x < m_source.getOrigin().x) // Direction to sensor is flipped. Compensate
////		dirToSensor.x = -dirToSensor.x;
//
//#ifdef PRINT_DEBUGLOG
//	std::cout << "dirToSensor: (" << dirToSensor.x << ", " << dirToSensor.y << ", " << dirToSensor.z << ") \n";
//#endif
//
//
//#ifndef OMEGA_TRACKING
//	dirToSensor *= getMediumIor(p1, scaling);
//#endif
//
//	Float distToSensor;
//	if(!movePhotonTillSensor(p1, dirToSensor, distToSensor, distTravelled, sampler, scaling))
//		return;
//
////#ifdef OMEGA_TRACKING
//	dirToSensor.normalize();
////#endif
//
//	VectorType<Float> refrDirToSensor = dirToSensor;
//	Float fresnelWeight = FPCONST(1.0);
//	Float ior = getMediumIor(p1, scaling);
//
//	if (ior > FPCONST(1.0)) {
//		refrDirToSensor.x = refrDirToSensor.x/ior;
//		refrDirToSensor.normalize();
//#ifdef PRINT_DEBUGLOG
//        std::cout << "refrDir: (" << refrDirToSensor[0] << ", " <<  refrDirToSensor[1] << ", " << refrDirToSensor[2] << ");" << std::endl;
//#endif
//#ifndef USE_NO_FRESNEL
//		fresnelWeight = (FPCONST(1.0) -
//		util::fresnelDielectric(dirToSensor.x, refrDirToSensor.x,
//			FPCONST(1.0) / ior))
//			/ ior / ior;
//#endif
//	}
//	Float foreshortening = dot(refrDirToSensor, m_camera.getDir())/dot(dirToSensor, m_camera.getDir());
//	Assert(foreshortening >= FPCONST(0.0));
//
//#if USE_SIMPLIFIED_TIMING
//	Float totalOpticalDistance = (distTravelled + distToSensor) * m_ior;
//#else
//	Float totalOpticalDistance = distTravelled;
//#endif
//
//	Float distanceToSensor = 0;
//	if(!m_camera.propagateTillSensor(p1, refrDirToSensor, distanceToSensor))
//		return;
//	totalOpticalDistance += distanceToSensor;
//
//	Float totalPhotonValue = val*(2*M_PI)
//			* std::exp(-medium.getSigmaT() * distToSensor)
//			* medium.getPhaseFunction()->f(d/d.length(), dirToSensor) // FIXME: Should be refractive index
//			* foreshortening
//			* fresnelWeight;
//	addEnergyToImage(img, p1, totalOpticalDistance, depth, totalPhotonValue);
//#ifdef PRINT_DEBUGLOG
//    std::cout << "Added Energy:" << totalPhotonValue << " to (" << p1.x << ", " << p1.y << ", " << p1.z << ") at time:" << totalOpticalDistance << std::endl;
//    std::cout << "val term:" << val << std::endl;
//    std::cout << "exp term:" << std::exp(-medium.getSigmaT() * distToSensor) << std::endl;
//    std::cout << "phase function term:" << medium.getPhaseFunction()->f(d/d.length(), dirToSensor) << std::endl;
//    std::cout << "fresnel weight:" << fresnelWeight << std::endl;
//#endif
}

// Move photon and return true if still in medium, false otherwise
__device__ bool Scene::movePhoton(TVector3<Float> &p, TVector3<Float> &d, Float dist,
                                  Float &totalOpticalDistance, short &uses, Float scaling) const{

	// Algorithm
	// 1. Move till you reach the boundary or till the distance is reached.
	// 2. If you reached the boundary, reflect with probability and keep progressing TODO: change to weight


	Float disx, disy;
	TVector3<Float> d1, norm;
	traceTillBlock(p, d, dist, disx, disy, totalOpticalDistance, scaling);

	dist -= disy;

	while(dist > M_EPSILON){
		int i;
		norm.zero();
		for (i = 0; i < p.dim; ++i) {
			if (fabsf(m_block->getBlockL()[i] - p[i]) < M_EPSILON) {
				norm[i] = -FPCONST(1.0);
				break;
			}
			else if (fabsf(m_block->getBlockR()[i] - p[i]) < M_EPSILON) {
				norm[i] = FPCONST(1.0);
				break;
			}
		}
		ASSERT(i < p.dim);

		Float minDiff = M_MAX;
		Float minDir = FPCONST(0.0);
		TVector3<Float> normalt;
		normalt.zero();
		int chosenI = p.dim;
		for (i = 0; i < p.dim; ++i) {
			Float diff = fabsf(m_block->getBlockL()[i] - p[i]);
			if (diff < minDiff) {
				minDiff = diff;
				chosenI = i;
				minDir = -FPCONST(1.0);
			}
			diff = fabsf(m_block->getBlockR()[i] - p[i]);
			if (diff < minDiff) {
				minDiff = diff;
				chosenI = i;
				minDir = FPCONST(1.0);
			}
		}
		normalt[chosenI] = minDir;
		ASSERT(normalt == norm);
		norm = normalt;

		/*
		 * TODO: I think that, because we always return to same medium (we ignore
		 * refraction), there is no need to adjust radiance by eta*eta.
		 */
		Float magnitude = d.length();
#ifdef PRINT_DEBUGLOG
		std::cout << "Before BSDF sample, d: (" << d.x/magnitude << ", " << d.y/magnitude <<  ", " << d.z/magnitude << "); \n "
				"norm: (" << norm.x << ", " << norm.y << ", " << norm.z << ");" << "A Sampler: " << sampler() << "\n";
#endif
        m_bsdf->sample(d/magnitude, norm, *sampler, d1, uses);
        if (dot(d1, norm) < FPCONST(0.0)) {
			// re-enter the medium through reflection
			d = d1*magnitude;
		} else {
			return false;
		}

    	traceTillBlock(p, d, dist, disx, disy, totalOpticalDistance, scaling);
    	dist -= disy;
	}
	return true;
}

__device__ bool scatterOnce(TVector3<Float> &p, TVector3<Float> &d, Float &dist,
						Float &totalOpticalDistance, const Float &scaling, short &samplerUses) {
    Medium *medium = d_constants.medium;
    Scene *scene = d_constants.scene;
    Sampler &sampler = *scene->sampler;

	if ((medium->getAlbedo() > FPCONST(0.0)) && ((medium->getAlbedo() >= FPCONST(1.0)) || (sampler(samplerUses) < medium->getAlbedo()))) {
		TVector3<Float> d1;
		Float magnitude = d.length();
		medium->getPhaseFunction()->sample(d/magnitude, sampler, samplerUses, d1);
		d = magnitude*d1;
		dist = getMoveStep(medium, samplerUses);
#ifdef PRINT_DEBUGLOG
		std::cout << "sampler before move photon:" << sampler() << "\n";
#endif
		return scene->movePhoton(p, d, dist, totalOpticalDistance, samplerUses, scaling);
	} else {
		dist = FPCONST(0.0);
		return false;
	}
}

__device__ void scatter(TVector3<Float> &p, TVector3<Float> &d, Float scaling, Float &totalOpticalDistance, short &uses) {
    Scene *scene = d_constants.scene;
    Medium *medium = d_constants.medium;
    Sampler &sampler = *scene->sampler;
	ASSERT(scene->getMediumBlock()->inside(p));

	if ((medium->getAlbedo() > FPCONST(0.0)) && ((medium->getAlbedo() >= FPCONST(1.0)) || (sampler(uses) < medium->getAlbedo()))) {
		TVector3<Float> pos(p), dir(d);

		Float dist = getMoveStep(medium, uses);
		if (!scene->movePhoton(pos, dir, dist, totalOpticalDistance, uses, scaling)) {
			return;
		}

		int depth = 1;
		Float totalDist = dist;
//		while ((m_maxDepth < 0 || depth <= m_maxDepth) &&
//				(m_maxPathlength < 0 || totalDist <= m_maxPathlength)) {
//          ASSERT(m_useAngularSampling);
//			if(m_useAngularSampling)
//              scene.addEnergyInParticle(img, pos, dir, totalOpticalDistance, depth, weight, medium, sampler, scaling);
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
	}
}

__global__ void renderPhotons() {
    //int i = threadIdx.x;
    //int num_threads = blockDim.x;

    // TODO: Zero out constants.image

    TVector3<Float> pos;
    TVector3<Float> dir;
    Float totalDistance;
    short uses = 0;

    Scene *scene = d_constants.scene;
    Sampler &sampler = *scene->sampler;

    if (scene->genRay(pos, dir, totalDistance, uses)) {
        float scaling = max(min(sinf(scene->getUSPhi_min() + scene->getUSPhi_range() * sampler(uses)), scene->getUSMaxScaling()), -scene->getUSMaxScaling());
        // TODO: Implement direct tracing, and check useDirect before using it
        // directTracing(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, totalDistance); // Traces and adds direct energy, which is equal to weight * exp( -u_t * path_length);
        scatter(pos, dir, scaling, totalDistance, uses);
    }
}

void CudaRenderer::renderImage(image::SmallImage& target, const med::Medium &medium, const scn::Scene<tvec::TVector3> &scene, int numPhotons) {
    setup(target, medium, scene, numPhotons);

    renderPhotons<<<1,numPhotons>>>();
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(image, cudaImage,
                         target.getXRes()*target.getYRes()*target.getZRes()*sizeof(Float),
                         cudaMemcpyDeviceToHost));

    target.copyImage(image, target.getXRes()*target.getYRes()*target.getZRes());

    cleanup();
}

/* Allocates host and device data and sets up RNG. */
//TODO: introduce medium
void CudaRenderer::setup(image::SmallImage& target, const med::Medium &medium, const scn::Scene<tvec::TVector3> &scene, int numPhotons) {
    /* Allocate host memory */
    image = new Float[target.getXRes()*target.getYRes()*target.getZRes()*sizeof(Float)];

    /* Allocate device memory*/
    CUDA_CALL(cudaMalloc((void **)&cudaImage,
                         target.getXRes()*target.getYRes()*target.getZRes()*sizeof(Float)));
    CUDA_CALL(cudaMalloc((void **)&cudaRandom, requiredRandomNumbers(numPhotons) * sizeof(Float)));
    cudaScene = Scene::from(scene, cudaRandom, requiredRandomNumbers(numPhotons) * sizeof(Float));
    cudaMedium = Medium::from(medium);

    /* Setup generator. */
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937));

    /* Send in parameter pointers to device */
    Constants h_constants;
    h_constants.image = cudaImage;
    h_constants.random = cudaRandom;
    h_constants.scene = cudaScene;
    h_constants.medium = cudaMedium;
    h_constants.weight = getWeight(medium, scene, numPhotons);

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

    // TODO: Free cudaImage, cudaRandom, cudaScene, cudaMedium
}

CudaRenderer::~CudaRenderer() {}

/* Required amount of random numbers to run the renderPhotons kernel on numPhotons */
unsigned int CudaRenderer::requiredRandomNumbers(int numPhotons) {
    return numPhotons * RANDOM_NUMBERS_PER_PHOTON;
}

}
