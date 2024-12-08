/*
    This file is adapated from Mitsuba
*/

#pragma once
#if !defined(__MITSUBA_CORE_WARP_H_)
#define __MITSUBA_CORE_WARP_H_

#include "util.h"
#include "math.h"
/**
 * \brief Implements common warping techniques that map from the unit
 * square to other domains, such as spheres, hemispheres, etc.
 *
 * The main application of this class is to generate uniformly
 * distributed or weighted point sets in certain common target domains.
 */
namespace warp {
	// =============================================================
	//! @{ \name Warping techniques related to spheres and subsets
	// =============================================================

	/// Uniformly sample a vector on the unit sphere with respect to solid angles
	extern tvec::Vec3f squareToUniformSphere(const tvec::Vec2f &sample);

	/// Density of \ref squareToUniformSphere() with respect to solid angles
	extern inline Float squareToUniformSpherePdf() { return INV_FOURPI; }

	/// Uniformly sample a vector on the unit hemisphere with respect to solid angles
	extern tvec::Vec3f squareToUniformHemisphere(const tvec::Vec2f &sample);

	/// Density of \ref squareToUniformHemisphere() with respect to solid angles
	extern inline Float squareToUniformHemispherePdf() { return INV_TWOPI; }

	/// Sample a cosine-weighted vector on the unit hemisphere with respect to solid angles
	extern tvec::Vec3f squareToCosineHemisphere(const tvec::Vec2f &sample);

	/// Density of \ref squareToCosineHemisphere() with respect to solid angles
	extern inline Float squareToCosineHemispherePdf(const tvec::Vec3f &d)
		{ return INV_PI * d.z; }

	/**
	 * \brief Uniformly sample a vector that lies within a given
	 * cone of angles around the Z axis
	 *
	 * \param cosCutoff Cosine of the cutoff angle
	 * \param sample A uniformly distributed sample on \f$[0,1]^2\f$
	 */
	extern tvec::Vec3f squareToUniformCone(Float cosCutoff, const tvec::Vec2f &sample);

	/**
	 * \brief Uniformly sample a vector that lies within a given
	 * cone of angles around the Z axis
	 *
	 * \param cosCutoff Cosine of the cutoff angle
	 * \param sample A uniformly distributed sample on \f$[0,1]^2\f$
	 */
	extern inline Float squareToUniformConePdf(Float cosCutoff) {
		return INV_TWOPI / (1-cosCutoff);
	}

	//! @}
	// =============================================================

	// =============================================================
	//! @{ \name Warping techniques that operate in the plane
	// =============================================================

	/// Uniformly sample a vector on a 2D disk
	extern tvec::Vec2f squareToUniformDisk(const tvec::Vec2f &sample);

	/// Density of \ref squareToUniformDisk per unit area
	extern inline Float squareToUniformDiskPdf() { return INV_PI; }

	/// Low-distortion concentric square to disk mapping by Peter Shirley (PDF: 1/PI)
	extern tvec::Vec2f squareToUniformDiskConcentric(const tvec::Vec2f &sample);

	/// Inverse of the mapping \ref squareToUniformDiskConcentric
	extern tvec::Vec2f uniformDiskToSquareConcentric(const tvec::Vec2f &p);

	/// Density of \ref squareToUniformDisk per unit area
	extern inline Float squareToUniformDiskConcentricPdf() { return INV_PI; }

	/// Convert an uniformly distributed square sample into barycentric coordinates
	extern tvec::Vec2f squareToUniformTriangle(const tvec::Vec2f &sample);

	/**
	 * \brief Sample a point on a 2D standard normal distribution
	 *
	 * Internally uses the Box-Muller transformation
	 */
	extern tvec::Vec2f squareToStdNormal(const tvec::Vec2f &sample);

	/// Density of \ref squareToStdNormal per unit area
	extern Float squareToStdNormalPdf(const tvec::Vec2f &pos);

	/// Warp a uniformly distributed square sample to a 2D tent distribution
	extern tvec::Vec2f squareToTent(const tvec::Vec2f &sample);

	/**
	 * \brief Warp a uniformly distributed sample on [0, 1] to a nonuniform
	 * tent distribution with nodes <tt>{a, b, c}</tt>
	 */
	extern Float intervalToNonuniformTent(Float a, Float b, Float c, Float sample);

	//! @}
	// =============================================================
}


#endif /* __MITSUBA_CORE_WARP_H_ */
