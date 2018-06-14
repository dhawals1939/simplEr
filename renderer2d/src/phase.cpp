/*
 * phase.cpp
 *
 *  Created on: Nov 23, 2015
 *      Author: igkiou
 */

#include <algorithm>
#include <fstream>
#include <stdio.h>
#include "phase.h"
#include "util.h"

namespace pfunc {

Float HenyeyGreenstein::f(const tvec::Vec3f &in, const tvec::Vec3f &out) const {
	Float cosTheta = tvec::dot(in, out);
	return f(cosTheta);
}

Float HenyeyGreenstein::f(Float cosTheta) const {
		return static_cast<Float>(FPCONST(1.0) / (2.0 * M_PI)) * (FPCONST(1.0) - m_g * m_g)
							/ (FPCONST(1.0) + m_g * m_g - FPCONST(2.0) * m_g * cosTheta);
}

Float HenyeyGreenstein::derivf(const tvec::Vec3f &in, const tvec::Vec3f &out) const {
	Float cosTheta = tvec::dot(in, out);
	Float denominator = FPCONST(1.0) + m_g * m_g - FPCONST(2.0) * m_g * cosTheta;
	return static_cast<Float>(FPCONST(1.0) / M_PI) *
						(cosTheta + cosTheta * m_g * m_g - FPCONST(2.0) * m_g)
						/ denominator / denominator;
}

Float HenyeyGreenstein::score(const tvec::Vec3f &in, const tvec::Vec3f &out) const {
	Float cosTheta = tvec::dot(in, out);
	return score(cosTheta);
}

Float HenyeyGreenstein::score(Float cosTheta) const {
	return (cosTheta + cosTheta * m_g * m_g - FPCONST(2.0) * m_g) * FPCONST(2.0)
			/ (FPCONST(1.0) - m_g * m_g)
			/ (FPCONST(1.0) + m_g * m_g - FPCONST(2.0) * m_g * cosTheta);
}

Float HenyeyGreenstein::sample(const tvec::Vec3f &in, smp::Sampler &sampler, tvec::Vec3f &out) const {

	Float sampleVal = FPCONST(1.0) - FPCONST(2.0) * sampler();

    Float theta;
    if (std::abs(m_g) < M_EPSILON) {
    	theta = M_PI * sampleVal;
	} else {
		theta = FPCONST(2.0) * std::atan((FPCONST(1.0) - m_g) / (FPCONST(1.0) + m_g)
							* std::tan(M_PI / FPCONST(2.0) * sampleVal));
	}
    Float cosTheta = std::cos(theta);
    Float sinTheta = std::sin(theta);

    tvec::Vec3f axisX, axisY;
    util::coordinateSystem(in, axisX, axisY);

    out = sinTheta * axisY + cosTheta * in;
    return cosTheta;
}

}	/* namespace pfunc */
