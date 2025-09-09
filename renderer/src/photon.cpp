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

#if USE_CUDA
#include "cuda_renderer.h"
#endif

namespace photon {

template <template <typename> class vector_type>
bool Renderer<vector_type>::scatterOnce(vector_type<Float> &p, vector_type<Float> &d, Float &dist,
						const scn::Scene<vector_type> &scene, const med::Medium &medium, Float &totalOpticalDistance,
						smp::Sampler &sampler, const Float &scaling) const {

	if ((medium.getAlbedo() > FPCONST(0.0)) && ((medium.getAlbedo() >= FPCONST(1.0)) || (sampler() < medium.getAlbedo()))) {
		vector_type<Float> d1;
		Float magnitude = d.length();
		medium.getPhaseFunction()->sample(d/magnitude, sampler, d1);
		d = magnitude*d1;
		dist = getMoveStep(medium, sampler);
#if PRINT_DEBUGLOG
		std::cout << "sampler before move photon:" << sampler() << "\n";
#endif
		return scene.movePhoton(p, d, dist, totalOpticalDistance, sampler, scaling);
	} else {
		dist = FPCONST(0.0);
		return false;
	}
}

template <template <typename> class vector_type>
void Renderer<vector_type>::directTracing(const vector_type<Float> &p, const vector_type<Float> &d,
					const scn::Scene<vector_type> &scene, const med::Medium &medium,
					smp::Sampler &sampler, image::SmallImage &img, Float weight, const Float &scaling, Float &totalOpticalDistance) const { // Adithya: Should this be in scene.cpp

	vector_type<Float> p1 = p;
	vector_type<Float> d1 = d;

	Float distToSensor;
	if(!scene.movePhotonTillSensor(p1, d1, distToSensor, totalOpticalDistance, sampler, scaling))
		return;
	Float fresnelWeight = FPCONST(1.0);

#ifndef OMEGA_TRACKING
	d1.normalize();
#endif
	Float ior = scene.get_medium_ior(p1, scaling);
	vector_type<Float> refrDirToSensor = d1;

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

	Float foreshortening = dot(refrDirToSensor, scene.get_camera().get_dir())/dot(d1, scene.get_camera().get_dir());
	Assert(foreshortening >= FPCONST(0.0));

#if USE_SIMPLIFIED_TIMING
	total_distance = (distToSensor) * ior;
#endif

	Float distanceToSensor = 0;
	if(!scene.get_camera().propagate_till_sensor(p1, refrDirToSensor, distanceToSensor))
		return;
	totalOpticalDistance += distanceToSensor;

	Float totalPhotonValue = weight
			* std::exp(-medium.getSigmaT() * distToSensor)
			* fresnelWeight;
	int depth = 0;
	scene.addEnergyToImage(img, p1, totalOpticalDistance, depth, totalPhotonValue);
}

template <template <typename> class vector_type>

void Renderer<vector_type>::scatter(const vector_type<Float> &p, const vector_type<Float> &d,
					const scn::Scene<vector_type> &scene, const med::Medium &medium,
					smp::Sampler &sampler, image::SmallImage &img, Float weight, const Float &scaling, Float &totalOpticalDistance) const {

	Assert(scene.get_medium_block().inside(p));

	if ((medium.getAlbedo() > FPCONST(0.0)) && ((medium.getAlbedo() >= FPCONST(1.0)) || (sampler() < medium.getAlbedo()))) {
		vector_type<Float> pos(p), dir(d);

		Float dist = getMoveStep(medium, sampler);

		if (!scene.movePhoton(pos, dir, dist, totalOpticalDistance, sampler, scaling)) {
			return;
		}

#if PRINT_DEBUGLOG
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
			if (!scatterOnce(pos, dir, dist, scene, medium, totalOpticalDistance, sampler, scaling)){
#if PRINT_DEBUGLOG
				std::cout << "sampler after failing scatter once:" << sampler() << std::endl;
#endif
				break;
			}
#if PRINT_DEBUGLOG
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

//template <template <typename> class vector_type>
//void Renderer<vector_type>::scatterDeriv(const vector_type<Float> &p, const vector_type<Float> &d,
//							const scn::Scene<vector_type> &scene, const med::Medium &medium,
//							smp::Sampler &sampler, image::SmallImage &img,
//							image::SmallImage &dSigmaT, image::SmallImage &dAlbedo,
//							image::SmallImage &dGVal, Float weight) const {
//
//	Assert(scene.get_medium_block().inside(p));
//
//	if ((medium.getAlbedo() > FPCONST(0.0)) && ((medium.getAlbedo() >= FPCONST(1.0)) || (sampler() < medium.getAlbedo()))) {
//		vector_type<Float> pos(p), dir(d);
//
//		Float dist = getMoveStep(medium, sampler);
//		if (!scene.movePhoton(pos, dir, dist, sampler)) {
//			return;
//		}
//
//		int depth = 1;
//		vector_type<Float> prevDir(d);
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

//template <template <typename> class vector_type>
//bool Renderer<vector_type>::scatterOnceWeight(vector_type<Float> &p, vector_type<Float> &d, Float &weight,
//						Float &dist, const scn::Scene<vector_type> &scene,
//						const med::Medium &medium, const med::Medium &samplingMedium,
//						smp::Sampler &sampler) const {
//
//	if ((samplingMedium.getAlbedo() > FPCONST(0.0)) && ((samplingMedium.getAlbedo() >= FPCONST(1.0))
//			|| (sampler() < samplingMedium.getAlbedo()))) {
//		vector_type<Float> d1;
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

//template <template <typename> class vector_type>
//void Renderer<vector_type>::scatterDerivWeight(const vector_type<Float> &p, const vector_type<Float> &d,
//							const scn::Scene<vector_type> &scene, const med::Medium &medium,
//							const med::Medium &samplingMedium,
//							smp::Sampler &sampler, image::SmallImage &img,
//							image::SmallImage &dSigmaT, image::SmallImage &dAlbedo,
//							image::SmallImage &dGVal, Float weight) const {
//
//	Assert(scene.get_medium_block().inside(p));
//
//	if ((samplingMedium.getAlbedo() > FPCONST(0.0)) && ((samplingMedium.getAlbedo() >= FPCONST(1.0)) ||
//		(sampler() < samplingMedium.getAlbedo()))) {
//		vector_type<Float> pos(p), dir(d);
//
//		Float dist = getMoveStep(samplingMedium, sampler);
//#if USE_PRINTING
//		std::cout << "sampled first = " << dist << std::endl;
//#endif
//		if (!scene.movePhoton(pos, dir, dist, sampler)) {
//			return;
//		}
//
//		int depth = 1;
//		vector_type<Float> prevDir(d);
//		Float totalDist = dist;
//		weight *= (medium.getAlbedo() / samplingMedium.getAlbedo()) *
//				((medium.getSigmaT() * std::exp(-medium.getSigmaT() * dist)) /
//					(samplingMedium.getSigmaT() * std::exp(-samplingMedium.getSigmaT() * dist)));
//		Float sumScoreSigmaT = (FPCONST(1.0) - medium.getSigmaT() * dist);
//		Float sumScoreAlbedo = FPCONST(1.0) / medium.getAlbedo();
//		Float sumScoreGVal = FPCONST(0.0);
//		while ((m_maxDepth < 0 || depth <= m_maxDepth) &&
//				(m_maxPathlength < 0 || totalDist <= m_maxPathlength)) {
//#if USE_PRINTING
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
//#if USE_PRINTING
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
template <template <typename> class vector_type>
void Renderer<vector_type>::renderImage(image::SmallImage &img0,
				const med::Medium &medium, const scn::Scene<vector_type> &scene,
				const int64 numPhotons) const {

#if USE_CUDA
	cuda::CudaRenderer cuRenderer = cuda::CudaRenderer(m_maxDepth, m_maxPathlength, m_useDirect, m_useAngularSampling);
	cuRenderer.renderImage(img0, medium, scene, numPhotons);
#else
#if USE_THREADED
	int numThreads = omp_get_num_procs();
	if(m_threads > 0)
		numThreads = std::min(m_threads, numThreads);
	omp_set_num_threads(numThreads);
#else
	int numThreads = 1;
#endif /* USE_THREADED */

#if NDEBUG
	std::cout << "numthreads = " << numThreads << std::endl;
	std::cout << "numphotons = " << numPhotons << std::endl;
#endif
	smp::SamplerSet sampler(numThreads);

	image::SmallImageSet img(img0.get_x_res(), img0.get_y_res(), img0.getZRes(), numThreads);
	img.zero();

	Float weight = getWeight(medium, scene, numPhotons);
#if USE_PRINTING
	Float Li = scene.get_area_source().get_Li();
	std::cout << "weight " << weight << " Li " << Li << std::endl;
#endif

#if USE_THREADED
	#pragma omp parallel for
#endif
	for (int64 omp_i = 0; omp_i < numPhotons; ++omp_i) {
#if USE_THREADED
		const int id = omp_get_thread_num();
#else
		const int id = 0;
#endif
#if PRINT_DEBUGLOG
		std::cout << "id:" << id << "\n";
		std::cout << "sampler:" << sampler[id]() << "\n";
#endif
		// FIXME: Remove
		sampler[id].seed(omp_i);
		vector_type<Float> pos, dir;
		Float total_distance = 0;
		if (scene.genRay(pos, dir, sampler[id], total_distance)) {
			/*
			 * TODO: Direct energy computation is not implemented.
			 */

#if PRINT_DEBUGLOG
			Float scaling = 1; //Hack to match the logs.
#else
			Float scaling = std::max(std::min(std::sin(scene.getUSPhi_min() + scene.getUSPhi_range()*sampler[id]()), scene.getUSMaxScaling()), -scene.getUSMaxScaling());
#endif


#ifndef OMEGA_TRACKING
			dir *= scene.get_medium_ior(pos, scaling);
#endif
			if(m_useDirect)
					directTracing(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, total_distance); // Traces and adds direct energy, which is equal to weight * exp( -u_t * path_length);
			scatter(pos, dir, scene, medium, sampler[id], img[id], weight, scaling, total_distance);
		}
	}

	img.mergeImages(img0);
//	delete[] problem;
#endif /* USE_CUDA */
}

//template <template <typename> class vector_type>
//void Renderer<vector_type>::renderDerivImage(image::SmallImage &img0, image::SmallImage &dSigmaT0,
//					image::SmallImage &dAlbedo0, image::SmallImage &dGVal0,
//					const med::Medium &medium, const scn::Scene<vector_type> &scene,
//					const int64 numPhotons) const {
//
//#if USE_THREADED
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
//	image::SmallImageSet img(img0.get_x_res(), img0.get_y_res(), img0.getZRes(), numThreads);
//	img.zero();
//
//	image::SmallImageSet dSigmaT(dSigmaT0.get_x_res(), dSigmaT0.get_y_res(), dSigmaT0.getZRes(), numThreads);
//	dSigmaT.zero();
//
//	image::SmallImageSet dAlbedo(dAlbedo0.get_x_res(), dAlbedo0.get_y_res(), dAlbedo0.getZRes(), numThreads);
//	dAlbedo.zero();
//
//	image::SmallImageSet dGVal(dGVal0.get_x_res(), dGVal0.get_y_res(), dGVal0.getZRes(), numThreads);
//	dGVal.zero();
//
//	Float weight = getWeight(medium, scene, numPhotons);
//#if USE_PRINTING
//	Float Li = scene.get_area_source().get_Li();
//	std::cout << "weight " << weight << " Li " << Li << std::endl;
//#endif
//
//#if USE_THREADED
//	#pragma omp parallel for
//#endif
//	for (int64 omp_i = 0; omp_i < numPhotons; ++omp_i) {
//
//#if USE_THREADED
//		const int id = omp_get_thread_num();
//#else
//		const int id = 0;
//#endif
//		vector_type<Float> pos, dir;
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
//template <template <typename> class vector_type>
//void Renderer<vector_type>::renderDerivImageWeight(image::SmallImage &img0, image::SmallImage &dSigmaT0,
//					image::SmallImage &dAlbedo0, image::SmallImage &dGVal0,
//					const med::Medium &medium, const med::Medium &samplingMedium,
//					const scn::Scene<vector_type> &scene, const int64 numPhotons) const {
//
//#if USE_THREADED
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
//	image::SmallImageSet img(img0.get_x_res(), img0.get_y_res(), img0.getZRes(), numThreads);
//	img.zero();
//
//	image::SmallImageSet dSigmaT(dSigmaT0.get_x_res(), dSigmaT0.get_y_res(), dSigmaT0.getZRes(), numThreads);
//	dSigmaT.zero();
//
//	image::SmallImageSet dAlbedo(dAlbedo0.get_x_res(), dAlbedo0.get_y_res(), dAlbedo0.getZRes(), numThreads);
//	dAlbedo.zero();
//
//	image::SmallImageSet dGVal(dGVal0.get_x_res(), dGVal0.get_y_res(), dGVal0.getZRes(), numThreads);
//	dGVal.zero();
//
//	Float weight = getWeight(medium, scene, numPhotons);
//#if USE_PRINTING
//	Float Li = scene.get_area_source().get_Li();
//	std::cout << "weight " << weight << " Li " << Li << std::endl;
//#endif
//
//#if USE_THREADED
//	#pragma omp parallel for
//#endif
//	for (int64 omp_i = 0; omp_i < numPhotons; ++omp_i) {
//
//#if USE_THREADED
//		const int id = omp_get_thread_num();
//#else
//		const int id = 0;
//#endif
//		vector_type<Float> pos, dir;
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
