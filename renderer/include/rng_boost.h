/*
 * rng_boost.h
 *
 *  Created on: Nov 28, 2015
 *      Author: igkiou
 */
#pragma once

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

#include "constants.h"

namespace rng {

typedef unsigned int BoostEngineSeedType;

class BoostEngine  {
public:
	BoostEngine()
		: m_rng(), \
		  m_distrib(), \
		  m_sampler(m_rng, m_distrib) {
		seed();
	}

	template <typename IndexType>
	explicit BoostEngine(const IndexType seedValue)
		: m_rng(), \
		  m_distrib(), \
		  m_sampler(m_rng, m_distrib) {
		seed(seedValue);
	}

	inline void seed(const BoostEngineSeedType seedValue
												= BoostEngineSeedType(5489)) {
		m_rng.seed(seedValue);
	}

	template <typename IndexType>
	inline void seed(const IndexType seedvalue) {
		seed(static_cast<BoostEngineSeedType>(seedvalue));
	}

	inline Float operator()() {
		return static_cast<Float>(m_sampler());
	}

private:
	typedef boost::mt19937 Engine;
	typedef boost::uniform_real<float> RealDistribution;
	typedef boost::variate_generator<Engine&, RealDistribution> Generator;

	Engine m_rng;
	RealDistribution m_distrib;
	Generator m_sampler;
};

} /* namespace rng */