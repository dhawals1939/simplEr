/*
 * bsdf.cpp
 *
 *  Created on: Nov 29, 2015
 *      Author: igkiou
 */

#include "bsdf.h"
#include "util.h"
#include "constants.h"

namespace bsdf {

/*
Float BSDF::eval(const tvec::Vec3f &in, const tvec::Vec3f &n, const tvec::Vec3f &out) const {
    return 0.0f;
}
*/

/*
 * TODO: Need to make sure that if this is ever used for refraction, then the
 * calling routine will apply the eta*eta radiance scaling. See Veach, page 141.
 */

template <template <typename> class VectorType>
void SmoothDielectric<VectorType>::sample(const VectorType<Float> &in, const VectorType<Float> &n,
                              smp::Sampler &sampler, VectorType<Float> &out) const {

	if (std::abs(m_ior1 - m_ior2) < M_EPSILON) {
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

		VectorType<Float> outT;
		if (!util::refract(in, n, eta, outT)) {
			// TIR
			out = outT;
		} else {
			VectorType<Float> outR;
			util::reflect(in, n, outR);

			Float cosI = tvec::absDot(n, in), cosT = tvec::absDot(n, outT);
			Float fresnelR = util::fresnelDielectric(cosI, cosT, eta);

			// return either refracted or reflected direction based on the Fresnel term
			out = (sampler() < fresnelR ? outR : outT);
		}
	}
}

template class SmoothDielectric<tvec::TVector2>;
template class SmoothDielectric<tvec::TVector3>;

} //namespace bsdf
