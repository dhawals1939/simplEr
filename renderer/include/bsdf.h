/*
 * bsdf.h
 *
 *  Created on: Nov 29, 2015
 *      Author: igkiou
 */

#pragma once

#include <string>
#include <vector>
#include <sampler.h>
#include <tvector.h>
#include <constants.h>

namespace bsdf {

template <template <typename> class vector_type>
class SmoothDielectric {
public:
	SmoothDielectric(Float ior1, Float ior2) :
		m_ior1(ior1),
		m_ior2(ior2) { }

	SmoothDielectric(const SmoothDielectric &in) {
		m_ior1 = in.m_ior1; m_ior2 = in.m_ior2;
	}

	void sample(const vector_type<Float> &in, const vector_type<Float> &n,
				smp::Sampler &sampler, vector_type<Float> &out) const;

	inline Float getIor1() const {
		return m_ior1;
	}

	inline Float getIor2() const {
		return m_ior2;
	}

private:
	Float m_ior1;
	Float m_ior2;
};

} //namespace bsdf

