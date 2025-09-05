/*
 * photon.h
 *
 *  Created on: Nov 21, 2015
 *      Author: igkiou
 */
#pragma once

#include <stdio.h>
#include <vector>

#include <constants.h>
#include <image.h>
#include <phase.h>
#include <sampler.h>
#include <tvector.h>
#include <medium.h>
#include <scene.h>

#include <omp.h>

namespace photon {

template <template <typename> class vector_type>
class Renderer {
public:
	Renderer(const int maxDepth, const Float maxPathlength, const bool useDirect, const bool useAngularSampling, const int64_t threads) :
			m_maxDepth(maxDepth),
			m_maxPathlength(maxPathlength),
			m_useDirect(useDirect),
			m_useAngularSampling(useAngularSampling),
			m_threads(threads){
#ifndef NDEBUG
		std::cout << "maxDepth " << m_maxDepth << std::endl;
		std::cout << "maxPathlength " << m_maxPathlength << std::endl;
		std::cout << "useDirect " << m_useDirect << std::endl;
		std::cout << "useAngularSampling " << m_useAngularSampling << std::endl;
		std::cout << "threads " << m_threads << std::endl;
#endif
	}

	bool scatterOnce(vector_type<Float> &p, vector_type<Float> &d, Float &dist,
					const scn::Scene<vector_type> &scene, const med::Medium &medium, Float &totalOpticalDistance,
					smp::Sampler &sampler, const Float &scaling) const;

	void directTracing(const vector_type<Float> &pos, const vector_type<Float> &dir,
					   const scn::Scene<vector_type> &scene, const med::Medium &medium,
					   smp::Sampler &sampler, image::SmallImage &img, Float weight, const Float &scaling, Float &totalOpticalDistance) const; // Traces and adds direct energy, which is equal to weight * exp( -u_t * path_length);

	void scatter(const vector_type<Float> &p, const vector_type<Float> &d,
				const scn::Scene<vector_type> &scene, const med::Medium &medium,
				smp::Sampler &sampler, image::SmallImage &img, Float weight, const Float &scaling, Float &totalOpticalDistance) const;

	void scatterDeriv(const vector_type<Float> &p, const vector_type<Float> &d,
					const scn::Scene<vector_type> &scene, const med::Medium &medium,
					smp::Sampler &sampler, image::SmallImage &img,
					image::SmallImage &dSigmaT, image::SmallImage &dAlbedo,
					image::SmallImage &dGVal, Float weight) const;

	bool scatterOnceWeight(vector_type<Float> &p, vector_type<Float> &d, Float &weight,
							Float &dist, const scn::Scene<vector_type> &scene,
							const med::Medium &medium, const med::Medium &samplingMedium,
							smp::Sampler &sampler) const;

	void scatterDerivWeight(const vector_type<Float> &p, const vector_type<Float> &d,
						const scn::Scene<vector_type> &scene, const med::Medium &medium,
						const med::Medium &samplingMedium,
						smp::Sampler &sampler, image::SmallImage &img,
						image::SmallImage &dSigmaT, image::SmallImage &dAlbedo,
						image::SmallImage &dGVal, Float weight) const;

	inline Float getMoveStep(const med::Medium &medium, smp::Sampler &sampler) const {
		return -medium.getMfp() * std::log(sampler());
	}

	static inline Float getWeight(const med::Medium &, const scn::Scene<vector_type> &scene,
						const int64 numPhotons) {
		return scene.getAreaSource().get_li() * scene.getFresnelTrans()
				/ static_cast<Float>(numPhotons);
	}

	void renderImage(image::SmallImage &img0,
					const med::Medium &medium, const scn::Scene<vector_type> &scene,
					const int64 numPhotons) const;

	void renderDerivImage(image::SmallImage &img0, image::SmallImage &dSigmaT0,
					image::SmallImage &dAlbedo0, image::SmallImage &dGVal0,
					const med::Medium &medium, const scn::Scene<vector_type> &scene,
					const int64 numPhotons) const;

	void renderDerivImageWeight(image::SmallImage &img0, image::SmallImage &dSigmaT0,
					image::SmallImage &dAlbedo0, image::SmallImage &dGVal0,
					const med::Medium &medium, const med::Medium &samplingMedium,
					const scn::Scene<vector_type> &scene, const int64 numPhotons) const;

	inline int getMaxDepth() const {
		return m_maxDepth;
	}

	inline Float getMaxPathlength() const {
		return m_maxPathlength;
	}

	inline bool getUseDirect() const {
		return m_useDirect;
	}

	~Renderer() { }

protected:
	int m_maxDepth;
	Float m_maxPathlength;
	bool m_useDirect;
	bool m_useAngularSampling;
	int m_threads;
};

}	/* namespace photon */