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

class henyey_greenstein {
public:
	henyey_greenstein(const Float g)
					: m_g(g) {	}

	~henyey_greenstein() { }

	template <template <typename> class vector_type>
	Float f(const vector_type<Float> &in, const vector_type<Float> &out) const;

	template <template <typename> class vector_type>
	Float derivf(const vector_type<Float> &in, const vector_type<Float> &out) const;

	template <template <typename> class vector_type>
	Float score(const vector_type<Float> &in, const vector_type<Float> &out) const;

	template <template <typename> class vector_type>
	Float sample(const vector_type<Float> &in, smp::Sampler &sampler,
					vector_type<Float> &out)  const;

	inline Float getG() const {
		return m_g;
	}

private:
	Float m_g;
};

}	/* namespace pfunc */