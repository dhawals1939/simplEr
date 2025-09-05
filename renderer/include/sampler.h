/*
 * sampler.h
 *
 *  Created on: Nov 24, 2015
 *      Author: igkiou
 */
#pragma once

#include <constants.h>

#if USE_SFMT
#include <rng_sse.h>
#else
#include <rng_boost.h>
#endif

namespace smp {

#if USE_SFMT
typedef rng::SSEEngineSeedType SeedType;
typedef rng::SSEEngine Sampler;
#else
typedef rng::BoostEngineSeedType SeedType;
typedef rng::BoostEngine Sampler;
#endif

class SamplerSet {
public:
	explicit SamplerSet(const int numSamplers);

	SamplerSet(const int numSamplers, const unsigned int seedValue);

	explicit SamplerSet(const std::vector<unsigned int>& seedVector);

	inline Sampler &operator[](int i) {
		return m_samplers[i];
	}

	inline const Sampler &operator[](int i) const {
		return m_samplers[i];
	}

	~SamplerSet();

protected:
	int m_numSamplers;
	Sampler* m_samplers;
};

}	/* namespace smp */