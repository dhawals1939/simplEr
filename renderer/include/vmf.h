/*
    This file is adapated from Mitsuba
*/
#pragma once
#if !defined(__MITSUBA_CORE_VMF_H_)
#define __MITSUBA_CORE_VMF_H_


#include <util.h>
#include <sampler.h>

/**
 * \brief Von Mises-Fisher distribution on the 2-sphere
 *
 * This is a basic implementation, which assumes that the
 * distribution is centered around the X-axis. All provided
 * functions are implemented in such a way that they avoid
 * issues with numerical overflow.
 *
 */
struct VonMisesFisherDistr {
public:
	/**
	 * \brief Create a new von Mises-Fisher distribution
	 * with the given concentration parameter
	 */
	explicit inline VonMisesFisherDistr(Float kappa = 0) : m_kappa(kappa) { }

	/// Return the concentration parameter kappa
	inline void setKappa(Float kappa) {
		m_kappa = kappa;
	}

	/// Return the concentration parameter kappa
	inline Float getKappa() const {
		return m_kappa;
	}

	/// Return the mean cosine of the distribution
	Float getMeanCosine() const;

	/// Evaluate the distribution for a given value of cos(theta)
	Float eval(Float cosTheta) const;

	/**
	 * \brief Generate a sample from this distribution
	 *
	 * \param sample
	 *     A uniformly distributed point on <tt>[0,1]^2</tt>
	 */
	tvec::Vec3f sample(const tvec::Vec2f &sample) const;

	void sample(const tvec::Vec3f &in, smp::Sampler &smp, tvec::Vec3f &out) const;
	/**
	 * \brief Compute an appropriate concentration parameter so that
	 * the associated vMF distribution takes on the value \c x at its peak
	 */
	static Float forPeakValue(Float x);

	/**
	 * \brief Estimate the vMF concentration parameter
	 * based on the length of the mean vector that is produced
	 * by simply averaging a set of sampled directions
	 *
	 * This is an unbiased estimator [Banerjee et al. 05]
	 */
	static Float forMeanLength(Float length);

	/**
	 * \brief Compute an concentration parameter that approximately
	 * corresponds to the spherical convolution of two vMF distributions.
	 *
	 * For details, see "Directional Statistics" by Mardia and Jupp, p.44
	 */
	static Float convolve(Float kappa1, Float kappa2);
private:
	Float m_kappa;
};


#endif /* __MITSUBA_CORE_VMF_H_ */
