/*
 * phase.h
 *
 *  Created on: Nov 24, 2015
 *      Author: igkiou
 */
#pragma once

#include <string>
#include <vector>
#include <sampler.h>
#include <tvector.h>
#include <constants.h>

namespace pfunc {

class HenyeyGreenstein {
public:
	HenyeyGreenstein(const Float g)
					: m_g(g) {	}

	~HenyeyGreenstein() { }

	template <template <typename> class VectorType>
	Float f(const VectorType<Float> &in, const VectorType<Float> &out) const;

	template <template <typename> class VectorType>
	Float derivf(const VectorType<Float> &in, const VectorType<Float> &out) const;

	template <template <typename> class VectorType>
	Float score(const VectorType<Float> &in, const VectorType<Float> &out) const;

	template <template <typename> class VectorType>
	Float sample(const VectorType<Float> &in, smp::Sampler &sampler,
					VectorType<Float> &out)  const;

	inline Float getG() const {
		return m_g;
	}

private:
	Float m_g;
};

}	/* namespace pfunc */