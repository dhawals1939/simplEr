/*
 * photon.cpp
 *
 *  Created on: Nov 21, 2015
 *      Author: igkiou
 */

#include "photon.h"
#include "util.h"
#include <iterator>
#include <iostream>

namespace photon {

template <template <typename> class VectorType>
bool Renderer<VectorType>::scatterOnce(VectorType<Float> &p, VectorType<Float> &d, Float &dist,
						const scn::Scene<VectorType> &scene, const med::Medium &medium, Float &totalOpticalDistance,
						smp::Sampler &sampler, const Float &scaling) const {

	if ((medium.getAlbedo() > FPCONST(0.0)) && ((medium.getAlbedo() >= FPCONST(1.0)) || (sampler() < medium.getAlbedo()))) {
		VectorType<Float> d1;
		Float magnitude = d.length();
		medium.getPhaseFunction()->sample(d/magnitude, sampler, d1);
		d = magnitude*d1;
		dist = getMoveStep(medium, sampler);
#ifdef PRINT_DEBUGLOG
		std::cout << "sampler before move photon:" << sampler() << "\n";
#endif
		return scene.movePhoton(p, d, dist, totalOpticalDistance, sampler, scaling);
	} else {
		dist = FPCONST(0.0);
		return false;
	}
}

template <template <typename> class VectorType>
void Renderer<VectorType>::directTracing(const VectorType<Float> &p, const VectorType<Float> &d,
					const scn::Scene<VectorType> &scene, const med::Medium &medium,
					smp::Sampler &sampler, image::SmallImage &img, Float weight, const Float &scaling, Float &totalOpticalDistance) const { // Adithya: Should this be in scene.cpp

	VectorType<Float> p1 = p;
	VectorType<Float> d1 = d;

	Float distToSensor;
	if(!scene.movePhotonTillSensor(p1, d1, distToSensor, totalOpticalDistance, sampler, scaling))
		return;
	Float fresnelWeight = FPCONST(1.0);

#ifndef OMEGA_TRACKING
	d1.normalize();
#endif
	Float ior = scene.getMediumIor(p1, scaling);
	VectorType<Float> refrDirToSensor = d1;

	if (ior > FPCONST(1.0)) {
		refrDirToSensor.x = refrDirToSensor.x/ior;
		refrDirToSensor.normalize();
#ifndef USE_NO_FRESNEL
		fresnelWeight = (FPCONST(1.0) -
		util::fresnelDielectric(d1.x, refrDirToSensor.x,
			FPCONST(1.0) / ior))
			/ ior / ior;
#endif
	}

	Float foreshortening = dot(refrDirToSensor, scene.getCamera().getDir())/dot(d1, scene.getCamera().getDir());
	Assert(foreshortening >= FPCONST(0.0));

#if USE_SIMPLIFIED_TIMING
	totalDistance = (distToSensor) * ior;
#endif

	Float distanceToSensor = 0;
	if(!scene.getCamera().propagateTillSensor(p1, refrDirToSensor, distanceToSensor))
		return;
	totalOpticalDistance += distanceToSensor;


	Float totalPhotonValue = weight
			* std::exp(-medium.getSigmaT() * distToSensor)
			* fresnelWeight;
	int depth = 0;
	scene.addEnergyToImage(img, p1, totalOpticalDistance, depth, totalPhotonValue);
}

template <template <typename> class VectorType>
void Renderer<VectorType>::scatter(const VectorType<Float> &p, const VectorType<Float> &d,
					const scn::Scene<VectorType> &scene, const med::Medium &medium,
					smp::Sampler &sampler, image::SmallImage &img, Float weight, const Float &scaling, Float &totalOpticalDistance,
					scn::NEECostFunction<VectorType> &costFunction, Problem &problem, Float *initialization) const {

	Assert(scene.getMediumBlock().inside(p));

	if ((medium.getAlbedo() > FPCONST(0.0)) && ((medium.getAlbedo() >= FPCONST(1.0)) || (sampler() < medium.getAlbedo()))) {
		VectorType<Float> pos(p), dir(d);

		Float dist = getMoveStep(medium, sampler);
		if (!scene.movePhoton(pos, dir, dist, totalOpticalDistance, sampler, scaling)) {
			return;
		}

#ifdef PRINT_DEBUGLOG
		std::cout << "dist: " << dist << "\n";
		std::cout << "pos: (" << pos.x << ", " << pos.y << ", " << pos.z << ") " << "\n";
		std::cout << "dir: (" << dir.x << ", " << dir.y << ", " << dir.z << ") " << "\n";
#endif
		int depth = 1;
		Float totalDist = dist;
		while ((m_maxDepth < 0 || depth <= m_maxDepth) &&
				(m_maxPathlength < 0 || totalDist <= m_maxPathlength)) {
			if(m_useAngularSampling)
                scene.addEnergyInParticle(img, pos, dir, totalOpticalDistance, depth, weight, medium, sampler, scaling);
			else
				scene.addEnergy(img, pos, dir, totalOpticalDistance, depth, weight, medium, sampler, scaling, costFunction, problem, initialization);
			if (!scatterOnce(pos, dir, dist, scene, medium, totalOpticalDistance, sampler, scaling)){
#ifdef PRINT_DEBUGLOG
				std::cout << "sampler after failing scatter once:" << sampler() << std::endl;
#endif
				break;
			}
#ifdef PRINT_DEBUGLOG
			std::cout << "sampler after succeeding scatter once:" << sampler() << std::endl;

			std::cout << "dist: " << dist << "\n";
			std::cout << "pos: (" << pos.x << ", " << pos.y << ", " << pos.z << ", " << "\n";
			std::cout << "dir: (" << dir.x << ", " << dir.y << ", " << dir.z << ", " << "\n";
#endif
#if USE_SIMPLIFIED_TIMING
			totalOpticalDistance += dist;
#endif
			++depth;
		}
	}
}

//template <template <typename> class VectorType>
//void Renderer<VectorType>::scatterDeriv(const VectorType<Float> &p, const VectorType<Float> &d,
//							const scn::Scene<VectorType> &scene, const med::Medium &medium,
//							smp::Sampler &sampler, image::SmallImage &img,
//							image::SmallImage &dSigmaT, image::SmallImage &dAlbedo,
//							image::SmallImage &dGVal, Float weight) const {
//
//	Assert(scene.getMediumBlock().inside(p));
//
//	if ((medium.getAlbedo() > FPCONST(0.0)) && ((medium.getAlbedo() >= FPCONST(1.0)) || (sampler() < medium.getAlbedo()))) {
//		VectorType<Float> pos(p), dir(d);
//
//		Float dist = getMoveStep(medium, sampler);
//		if (!scene.movePhoton(pos, dir, dist, sampler)) {
//			return;
//		}
//
//		int depth = 1;
//		VectorType<Float> prevDir(d);
//		Float totalDist = dist;
//		Float sumScoreSigmaT = (FPCONST(1.0) - medium.getSigmaT() * dist);
//		Float sumScoreAlbedo = FPCONST(1.0) / medium.getAlbedo();
//		Float sumScoreGVal = FPCONST(0.0);
//		while ((m_maxDepth < 0 || depth <= m_maxDepth) &&
//				(m_maxPathlength < 0 || totalDist <= m_maxPathlength)) {
//			scene.addEnergyDeriv(img, dSigmaT, dAlbedo, dGVal, pos, dir,
//							totalDist, weight, sumScoreSigmaT, sumScoreAlbedo,
//							sumScoreGVal, medium, sampler);
//
//			prevDir = dir;
//			if (!scatterOnce(pos, dir, dist, scene, medium, sampler)) {
//				break;
//			}
//			totalDist += dist;
//			++depth;
//			sumScoreAlbedo += FPCONST(1.0) / medium.getAlbedo();
//			sumScoreSigmaT += (FPCONST(1.0) - medium.getSigmaT() * dist);
//			sumScoreGVal += medium.getPhaseFunction()->score(prevDir, dir);
//		}
//	}
//}

//template <template <typename> class VectorType>
//bool Renderer<VectorType>::scatterOnceWeight(VectorType<Float> &p, VectorType<Float> &d, Float &weight,
//						Float &dist, const scn::Scene<VectorType> &scene,
//						const med::Medium &medium, const med::Medium &samplingMedium,
//						smp::Sampler &sampler) const {
//
//	if ((samplingMedium.getAlbedo() > FPCONST(0.0)) && ((samplingMedium.getAlbedo() >= FPCONST(1.0))
//			|| (sampler() < samplingMedium.getAlbedo()))) {
//		VectorType<Float> d1;
//		samplingMedium.getPhaseFunction()->sample(d, sampler, d1);
//		dist = getMoveStep(samplingMedium, sampler);
//
//		weight *= (medium.getAlbedo() / samplingMedium.getAlbedo()) *
//				((medium.getSigmaT() * std::exp(-medium.getSigmaT() * dist)) /
//						(samplingMedium.getSigmaT() * std::exp(-samplingMedium.getSigmaT() * dist))) *
//				(medium.getPhaseFunction()->f(d, d1) / samplingMedium.getPhaseFunction()->f(d, d1));
//		d = d1;
//		return scene.movePhoton(p, d, dist, sampler);
//	} else {
//		dist = FPCONST(0.0);
//		return false;
//	}
//}

//template <template <typename> class VectorType>
//void Renderer<VectorType>::scatterDerivWeight(const VectorType<Float> &p, const VectorType<Float> &d,
//							const scn::Scene<VectorType> &scene, const med::Medium &medium,
//							const med::Medium &samplingMedium,
//							smp::Sampler &sampler, image::SmallImage &img,
//							image::SmallImage &dSigmaT, image::SmallImage &dAlbedo,
//							image::SmallImage &dGVal, Float weight) const {
//
//	Assert(scene.getMediumBlock().inside(p));
//
//	if ((samplingMedium.getAlbedo() > FPCONST(0.0)) && ((samplingMedium.getAlbedo() >= FPCONST(1.0)) ||
//		(sampler() < samplingMedium.getAlbedo()))) {
//		VectorType<Float> pos(p), dir(d);
//
//		Float dist = getMoveStep(samplingMedium, sampler);
//#ifdef USE_PRINTING
//		std::cout << "sampled first = " << dist << std::endl;
//#endif
//		if (!scene.movePhoton(pos, dir, dist, sampler)) {
//			return;
//		}
//
//		int depth = 1;
//		VectorType<Float> prevDir(d);
//		Float totalDist = dist;
//		weight *= (medium.getAlbedo() / samplingMedium.getAlbedo()) *
//				((medium.getSigmaT() * std::exp(-medium.getSigmaT() * dist)) /
//					(samplingMedium.getSigmaT() * std::exp(-samplingMedium.getSigmaT() * dist)));
//		Float sumScoreSigmaT = (FPCONST(1.0) - medium.getSigmaT() * dist);
//		Float sumScoreAlbedo = FPCONST(1.0) / medium.getAlbedo();
//		Float sumScoreGVal = FPCONST(0.0);
//		while ((m_maxDepth < 0 || depth <= m_maxDepth) &&
//				(m_maxPathlength < 0 || totalDist <= m_maxPathlength)) {
//#ifdef USE_PRINTING
//		std::cout << "total to add = " << totalDist << std::endl;
//#endif
////			if (depth == 2) {
//			scene.addEnergyDeriv(img, dSigmaT, dAlbedo, dGVal, pos, dir,
//							totalDist, weight, sumScoreSigmaT, sumScoreAlbedo,
//							sumScoreGVal, medium, sampler);
////			}
//
//			prevDir = dir;
//			if (!scatterOnceWeight(pos, dir, weight, dist, scene,
//								medium, samplingMedium, sampler)) {
//				break;
//			}
//#ifdef USE_PRINTING
//			std::cout << "sampled = " << dist << std::endl;
//#endif
//			totalDist += dist;
//			++depth;
//			sumScoreAlbedo += FPCONST(1.0) / medium.getAlbedo();
//			sumScoreSigmaT += (FPCONST(1.0) - medium.getSigmaT() * dist);
//			sumScoreGVal += medium.getPhaseFunction()->score(prevDir, dir);
//		}
//	}
//}

/**
 * Render an image.
 **/
template <template <typename> class VectorType>
void Renderer<VectorType>::renderImage(image::SmallImage &img0,
				const med::Medium &medium, const scn::Scene<VectorType> &scene,
				const int64 numPhotons) const {

#ifdef USE_THREADED
	int numThreads = omp_get_num_procs();
	if(m_threads > 0)
		numThreads = std::min(m_threads, numThreads);
	omp_set_num_threads(numThreads);
#else
	int numThreads = 1;
#endif
	/* Set-up least squares problem for doing next event estimation */
	Problem *problem = new Problem[numThreads];
	scn::NEECostFunction<tvec::TVector3> *costFunctions[numThreads];
	Float *initializations = new Float[numThreads*3]; // The initial parameter values (3 dimensional for x,y,z)

	for(int i=0; i<numThreads; i++){
		costFunctions[i] = new scn::NEECostFunction<tvec::TVector3>(&scene);
		problem[i].AddResidualBlock((CostFunction*)costFunctions[i], NULL, initializations +i*3);
	}


#ifndef NDEBUG
	std::cout << "numthreads = " << numThreads << std::endl;
	std::cout << "numphotons = " << numPhotons << std::endl;
#endif

	smp::SamplerSet sampler(numThreads);

	image::SmallImageSet img(img0.getXRes(), img0.getYRes(), img0.getZRes(), numThreads);
	img.zero();

	Float weight = getWeight(medium, scene, numPhotons);
#ifdef USE_PRINTING
	Float Li = scene.getAreaSource().getLi();
	std::cout << "weight " << weight << " Li " << Li << std::endl;
#endif

#ifdef USE_THREADED
	#pragma omp parallel for
#endif
	for (int64 omp_i = 0; omp_i < numPhotons; ++omp_i) {
#ifdef USE_THREADED
		const int id = omp_get_thread_num();
#else
		const int id = 0;
#endif
#ifdef PRINT_DEBUGLOG
		std::cout << "id:" << id << "\n";
		std::cout << "sampler:" << sampler[id]() << "\n";
#endif
		VectorType<Float> pos, dir;
		Float totalDistance = 0;
		if (scene.genRay(pos, dir, sampler[id], totalDistance)) {

			/*
			 * TODO: Direct energy computation is not implemented.
			 */
#ifdef PRINT_DEBUGLOG
			std::cout << "Intial pos: (" << pos.x << ", " << pos.y << ", " << pos.z << ") \n";
			std::cout << "Intial dir: (" << dir.x << ", " << dir.y << ", " << dir.z << ") \n";
#endif

#ifdef PRINT_DEBUGLOG
		Float scaling = 1; //Hack to match the logs.
#else
		Float scaling = std::max(std::min(std::sin(scene.getUSPhi_min() + scene.getUSPhi_range()*sampler[id]()), scene.getUSMaxScaling()), -scene.getUSMaxScaling());
#endif

#ifndef OMEGA_TRACKING
			dir *= scene.getMediumIor(pos, scaling);
#endif
			Assert(!m_useDirect);
			if(m_useDirect)
				directTracing(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, totalDistance); // Traces and adds direct energy, which is equal to weight * exp( -u_t * path_length);
			scatter(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, totalDistance, *costFunctions[id], problem[id], initializations+id*3);
		}
	}

	img.mergeImages(img0);


	for(int i=0; i<numThreads; i++){
		delete costFunctions[i];
	}
	delete[] initializations;
//	delete[] problem;
}

//template <template <typename> class VectorType>
//void Renderer<VectorType>::renderDerivImage(image::SmallImage &img0, image::SmallImage &dSigmaT0,
//					image::SmallImage &dAlbedo0, image::SmallImage &dGVal0,
//					const med::Medium &medium, const scn::Scene<VectorType> &scene,
//					const int64 numPhotons) const {
//
//#ifdef USE_THREADED
//	int numThreads = omp_get_num_procs();
//	omp_set_num_threads(numThreads);
//#else
//	int numThreads = 1;
//#endif
//#ifndef NDEBUG
//	std::cout << "numthreads = " << numThreads << std::endl;
//	std::cout << "numphotons = " << numPhotons << std::endl;
//#endif
//
//	smp::SamplerSet sampler(numThreads);
//
//	image::SmallImageSet img(img0.getXRes(), img0.getYRes(), img0.getZRes(), numThreads);
//	img.zero();
//
//	image::SmallImageSet dSigmaT(dSigmaT0.getXRes(), dSigmaT0.getYRes(), dSigmaT0.getZRes(), numThreads);
//	dSigmaT.zero();
//
//	image::SmallImageSet dAlbedo(dAlbedo0.getXRes(), dAlbedo0.getYRes(), dAlbedo0.getZRes(), numThreads);
//	dAlbedo.zero();
//
//	image::SmallImageSet dGVal(dGVal0.getXRes(), dGVal0.getYRes(), dGVal0.getZRes(), numThreads);
//	dGVal.zero();
//
//	Float weight = getWeight(medium, scene, numPhotons);
//#ifdef USE_PRINTING
//	Float Li = scene.getAreaSource().getLi();
//	std::cout << "weight " << weight << " Li " << Li << std::endl;
//#endif
//
//#ifdef USE_THREADED
//	#pragma omp parallel for
//#endif
//	for (int64 omp_i = 0; omp_i < numPhotons; ++omp_i) {
//
//#ifdef USE_THREADED
//		const int id = omp_get_thread_num();
//#else
//		const int id = 0;
//#endif
//		VectorType<Float> pos, dir;
//		if (scene.genRay(pos, dir, sampler[id])) {
//
//			/*
//			 * TODO: Direct energy computation is not implemented.
//			 */
//			Assert(!m_useDirect);
//			scatterDeriv(pos, dir, scene, medium, sampler[id], img[id],
//						dSigmaT[id], dAlbedo[id], dGVal[id], weight);
//		}
//	}
//
//	img.mergeImages(img0);
//	dSigmaT.mergeImages(dSigmaT0);
//	dAlbedo.mergeImages(dAlbedo0);
//	dGVal.mergeImages(dGVal0);
//}
//
//template <template <typename> class VectorType>
//void Renderer<VectorType>::renderDerivImageWeight(image::SmallImage &img0, image::SmallImage &dSigmaT0,
//					image::SmallImage &dAlbedo0, image::SmallImage &dGVal0,
//					const med::Medium &medium, const med::Medium &samplingMedium,
//					const scn::Scene<VectorType> &scene, const int64 numPhotons) const {
//
//#ifdef USE_THREADED
//	int numThreads = omp_get_num_procs();
//	omp_set_num_threads(numThreads);
//#else
//	int numThreads = 1;
//#endif
//#ifndef NDEBUG
//	std::cout << "numthreads = " << numThreads << std::endl;
//	std::cout << "numphotons = " << numPhotons << std::endl;
//#endif
//
//	smp::SamplerSet sampler(numThreads);
//
//	image::SmallImageSet img(img0.getXRes(), img0.getYRes(), img0.getZRes(), numThreads);
//	img.zero();
//
//	image::SmallImageSet dSigmaT(dSigmaT0.getXRes(), dSigmaT0.getYRes(), dSigmaT0.getZRes(), numThreads);
//	dSigmaT.zero();
//
//	image::SmallImageSet dAlbedo(dAlbedo0.getXRes(), dAlbedo0.getYRes(), dAlbedo0.getZRes(), numThreads);
//	dAlbedo.zero();
//
//	image::SmallImageSet dGVal(dGVal0.getXRes(), dGVal0.getYRes(), dGVal0.getZRes(), numThreads);
//	dGVal.zero();
//
//	Float weight = getWeight(medium, scene, numPhotons);
//#ifdef USE_PRINTING
//	Float Li = scene.getAreaSource().getLi();
//	std::cout << "weight " << weight << " Li " << Li << std::endl;
//#endif
//
//#ifdef USE_THREADED
//	#pragma omp parallel for
//#endif
//	for (int64 omp_i = 0; omp_i < numPhotons; ++omp_i) {
//
//#ifdef USE_THREADED
//		const int id = omp_get_thread_num();
//#else
//		const int id = 0;
//#endif
//		VectorType<Float> pos, dir;
//		if (scene.genRay(pos, dir, sampler[id])) {
//
//			/*
//			 * TODO: Direct energy computation is not implemented.
//			 */
//			Assert(!m_useDirect);
//			scatterDerivWeight(pos, dir, scene, medium, samplingMedium, sampler[id], img[id],
//						dSigmaT[id], dAlbedo[id], dGVal[id], weight);
//		}
//	}
//
//	img.mergeImages(img0);
//	dSigmaT.mergeImages(dSigmaT0);
//	dAlbedo.mergeImages(dAlbedo0);
//	dGVal.mergeImages(dGVal0);
//}

//template class Renderer<tvec::TVector2>;
template class Renderer<tvec::TVector3>;

}	/* namespace photon */
