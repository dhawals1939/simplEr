/*
    This file is adapated from Mitsuba
*/

#include "warp.h"


namespace warp {

tvec::Vec3f squareToUniformSphere(const tvec::Vec2f &sample) {
	Float z = 1.0f - 2.0f * sample.y;
	Float r = std::sqrt(1.0f - z*z);
	Float sinPhi, cosPhi;
#ifdef USE_DOUBLE_PRECISION
	sincos(2.0f * M_PI * sample.x, &sinPhi, &cosPhi);
#else
	sincosf(2.0f * M_PI * sample.x, &sinPhi, &cosPhi);
#endif /* USE_DOUBLE_PRECISION */
	return tvec::Vec3f(r * cosPhi, r * sinPhi, z);
}

tvec::Vec3f squareToUniformHemisphere(const tvec::Vec2f &sample) {
	Float z = sample.x;
	Float tmp = std::sqrt(1.0f - z*z);

	Float sinPhi, cosPhi;
#ifdef USE_DOUBLE_PRECISION
	sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);
#else
	sincosf(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);
#endif /*USE_DOUBLE_PRECISION*/

	return tvec::Vec3f(cosPhi * tmp, sinPhi * tmp, z);
}

tvec::Vec3f squareToCosineHemisphere(const tvec::Vec2f &sample) {
	tvec::Vec2f p = squareToUniformDiskConcentric(sample);
	Float z = std::sqrt(1.0f - p.x*p.x - p.y*p.y);

	/* Guard against numerical imprecisions */
	if (z == 0)
		z = 1e-10f;

	return tvec::Vec3f(p.x, p.y, z);
}

tvec::Vec3f squareToUniformCone(Float cosCutoff, const tvec::Vec2f &sample) {
	Float cosTheta = (1-sample.x) + sample.x * cosCutoff;
	Float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);

	Float sinPhi, cosPhi;
#ifdef USE_DOUBLE_PRECISION
	sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);
#else
	sincosf(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);
#endif /*USE_DOUBLE_PRECISION*/

	return tvec::Vec3f(cosPhi * sinTheta,
		sinPhi * sinTheta, cosTheta);
}

tvec::Vec2f squareToUniformDisk(const tvec::Vec2f &sample) {
	Float r = std::sqrt(sample.x);
	Float sinPhi, cosPhi;
#ifdef USE_DOUBLE_PRECISION
	sincos(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);
#else
	sincosf(2.0f * M_PI * sample.y, &sinPhi, &cosPhi);
#endif /*USE_DOUBLE_PRECISION*/

	return tvec::Vec2f(
		cosPhi * r,
		sinPhi * r
	);
}

tvec::Vec2f squareToUniformTriangle(const tvec::Vec2f &sample) {
	Float a = std::sqrt(1.0f - sample.x);
	return tvec::Vec2f(1 - a, a * sample.y);
}

tvec::Vec2f squareToUniformDiskConcentric(const tvec::Vec2f &sample) {
	Float r1 = 2.0f*sample.x - 1.0f;
	Float r2 = 2.0f*sample.y - 1.0f;

	/* Modified concencric map code with less branching (by Dave Cline), see
	   http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
	Float phi, r;
	if (r1 == 0 && r2 == 0) {
		r = phi = 0;
	} else if (r1*r1 > r2*r2) {
		r = r1;
		phi = (M_PI/4.0f) * (r2/r1);
	} else {
		r = r2;
		phi = (M_PI/2.0f) - (r1/r2) * (M_PI/4.0f);
	}

	Float cosPhi, sinPhi;
#ifdef USE_DOUBLE_PRECISION
	sincos(phi, &sinPhi, &cosPhi);
#else
	sincosf(phi, &sinPhi, &cosPhi);
#endif /*USE_DOUBLE_PRECISION*/

	return tvec::Vec2f(r * cosPhi, r * sinPhi);
}

tvec::Vec2f uniformDiskToSquareConcentric(const tvec::Vec2f &p) {
	Float r   = std::sqrt(p.x * p.x + p.y * p.y),
		  phi = std::atan2(p.y, p.x),
		  a, b;

	if (phi < -M_PI/4) {
  		/* in range [-pi/4,7pi/4] */
		phi += 2*M_PI;
	}

	if (phi < M_PI/4) { /* region 1 */
		a = r;
		b = phi * a / (M_PI/4);
	} else if (phi < 3*M_PI/4) { /* region 2 */
		b = r;
		a = -(phi - M_PI/2) * b / (M_PI/4);
	} else if (phi < 5*M_PI/4) { /* region 3 */
		a = -r;
		b = (phi - M_PI) * a / (M_PI/4);
	} else { /* region 4 */
		b = -r;
		a = -(phi - 3*M_PI/2) * b / (M_PI/4);
	}

	return tvec::Vec2f(0.5f * (a+1), 0.5f * (b+1));
}

tvec::Vec2f squareToStdNormal(const tvec::Vec2f &sample) {
	Float r   = std::sqrt(-2 * std::log(1-sample.x)),
		  phi = 2 * M_PI * sample.y;
	tvec::Vec2f result;
#ifdef USE_DOUBLE_PRECISION
	sincos(phi, &result.y, &result.x);
#else
	sincosf(phi, &result.y, &result.x);
#endif /*USE_DOUBLE_PRECISION*/
	return result * r;
}

Float squareToStdNormalPdf(const tvec::Vec2f &pos) {
	return INV_TWOPI * std::exp(-(pos.x*pos.x + pos.y*pos.y)/2.0f);
}

static Float intervalToTent(Float sample) {
	Float sign;

	if (sample < 0.5f) {
		sign = 1;
		sample *= 2;
	} else {
		sign = -1;
		sample = 2 * (sample - 0.5f);
	}

	return sign * (1 - std::sqrt(sample));
}

tvec::Vec2f squareToTent(const tvec::Vec2f &sample) {
	return tvec::Vec2f(
		intervalToTent(sample.x),
		intervalToTent(sample.y)
	);
}

Float intervalToNonuniformTent(Float a, Float b, Float c, Float sample) {
	Float factor;

	if (sample * (c-a) < b-a) {
		factor = a-b;
		sample *= (a-c)/(a-b);
	} else {
		factor = c-b;
		sample = (a-c)/(b-c) * (sample - (a-b)/(a-c));
	}

	return b + factor * (1-std::sqrt(sample));
}

}

