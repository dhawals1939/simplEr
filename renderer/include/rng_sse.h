/*
 * rng_sse.h
 *
 *  Created on: Nov 24, 2015
 *      Author: igkiou
 */

#ifndef RNG_SSE_H_
#define RNG_SSE_H_

#include <stdint.h>

#include "constants.h"

namespace rng {

typedef uint64_t SSEEngineSeedType;

class SSEEngine {
public:
	SSEEngine();

	template <typename IndexType>
	explicit SSEEngine(const IndexType seedValue);

	void seed(const SSEEngineSeedType seedValue = SSEEngineSeedType(5489));

	template <typename IndexType>
	inline void seed(const IndexType seedValue) {
		// TODO: Undo this change
		prev = seedValue;
		//seed(static_cast<SSEEngineSeedType>(seedValue));
	}

	inline Float operator()() {
		// TODO: REMOVE
		Float val = ((Float) nextULong() / (Float) 0x01000000);
 	    //printf("sampled %.4f\n", val);
        return val;
		// TODO: Enable
//#ifdef USE_DOUBLE_PRECISION
//		/* Trick from MTGP: generate an uniformly distributed
//		   single precision number in [1,2) and subtract 1. */
//		union {
//			uint64_t u;
//			double d;
//		} x;
//		x.u = (nextULong() >> 12) | 0x3ff0000000000000ULL;
// 	    printf("sampled %.2f\n", x.d - 1.0);
//		return x.d - 1.0;
//#else
//		/* Trick from MTGP: generate an uniformly distributed
//		   single precision number in [1,2) and subtract 1. */
//		union {
//			uint64_t u;
//			double d;
//		} x;
//		x.u = (nextULong() >> 12) | 0x3ff0000000000000ULL;
//		return static_cast<float>(x.d - 1.0);
//#endif
	}

	~SSEEngine();

private:
	/// Return an integer on the [0, 2^63-1]-interval
	uint64_t nextULong();

	/// Return an integer on the [0, n)-interval
	uint32_t nextUInt(uint32_t n);

	/// Return an integer on the [0, n)-interval
	size_t nextSize(size_t n);

	/// Return a floating point value on the [0, 1) interval
	Float nextFloat();

	/// Return a normally distributed value
	Float nextStandardNormal();

	struct State;
	State *mt;
	// TODO: REMOVE
	unsigned int prev;
};

} /* namespace rng */

#endif /* RNG_SSE_H_ */
