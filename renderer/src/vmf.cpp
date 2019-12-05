/*
    This file is adapated from Mitsuba
*/

#include "vmf.h"
#include "warp.h"

Float VonMisesFisherDistr::eval(Float cosTheta) const {
	if (m_kappa == 0.0f)
		return INV_FOURPI;
#if 0
	return exp(cosTheta * m_kappa)
		* m_kappa / (4 * M_PI * std::sinh(m_kappa));
#else
	/* Numerically stable version */
	return exp(m_kappa * std::min((Float)0, cosTheta - 1))
		* m_kappa / (2 * M_PI * (1-exp(-2*m_kappa)));
#endif
}

void VonMisesFisherDistr::sample(const tvec::Vec3f &in, smp::Sampler &smp, tvec::Vec3f &out) const {
    tvec::Vec3f axisX, axisY;
    util::coordinateSystem(in, axisX, axisY);

    tvec::Vec2f sample(smp(), smp());
	if (m_kappa < M_EPSILON){
	    out = warp::squareToUniformSphere(sample);
		return;
	}

#if 0
	Float cosTheta = log(exp(-m_kappa) + 2 *
						sample.x * std::sinh(m_kappa)) / m_kappa;
#else
	/* Numerically stable version */
	Float cosTheta = 1 + (log(sample.x +
		exp(-2 * m_kappa) * (1 - sample.x))) / m_kappa;
#endif

	Float sinTheta = std::sqrt(1-cosTheta*cosTheta),
	      sinPhi, cosPhi;

	sincos(2*M_PI * sample.y, &sinPhi, &cosPhi);

    out = (sinTheta * cosPhi) * axisX + (sinTheta * sinPhi) * axisY + cosTheta * in;
}

tvec::Vec3f VonMisesFisherDistr::sample(const tvec::Vec2f &sample) const {
	if (m_kappa == 0)
		return warp::squareToUniformSphere(sample);

#if 0
	Float cosTheta = log(exp(-m_kappa) + 2 *
						sample.x * std::sinh(m_kappa)) / m_kappa;
#else
	/* Numerically stable version */
	Float cosTheta = 1 + (log(sample.x +
		exp(-2 * m_kappa) * (1 - sample.x))) / m_kappa;
#endif

	Float sinTheta = std::sqrt(1-cosTheta*cosTheta),
	      sinPhi, cosPhi;

	sincos(2*M_PI * sample.y, &sinPhi, &cosPhi);

	return tvec::Vec3f(cosPhi * sinTheta,
		sinPhi * sinTheta, cosTheta);
}

Float VonMisesFisherDistr::getMeanCosine() const {
	if (m_kappa == 0)
		return 0;
	Float coth = m_kappa > 6 ? 1 : ((std::exp(2*m_kappa)+1)/(std::exp(2*m_kappa)-1));
	return coth-1/m_kappa;
}

static Float A3(Float kappa) {
	return 1/ std::tanh(kappa) - 1 / kappa;
}

static Float dA3(Float kappa) {
	Float csch = 2.0f /
		(exp(kappa)-exp(-kappa));
	return 1/(kappa*kappa) - csch*csch;
}

static Float A3inv(Float y, Float guess) {
	Float x = guess;
	int it = 1;

	while (true) {
		Float residual = A3(x)-y,
			  deriv = dA3(x);
		x -= residual/deriv;

		if (++it > 20) {
			AssertEx(false, "VanMisesFisherDistr::convolve(): Newton's method "
				" did not converge!");
			return guess;
		}

		if (std::abs(residual) < 1e-5f)
			break;
	}
	return x;
}

Float VonMisesFisherDistr::convolve(Float kappa1, Float kappa2) {
	return A3inv(A3(kappa1) * A3(kappa2), std::min(kappa1, kappa2));
}

Float VonMisesFisherDistr::forPeakValue(Float x) {
	if (x < INV_FOURPI) {
		return 0.0f;
	} else if (x > 0.795) {
		return 2 * M_PI * x;
	} else {
		return std::max((Float) 0.0f,
			(168.479f * x * x + 16.4585f * x - 2.39942f) /
			(-1.12718f * x * x + 29.1433f * x + 1.0f));
	}
}

static Float meanCosineFunctor(Float kappa, Float g) {
	return VonMisesFisherDistr(kappa).getMeanCosine()-g;
}

Float VonMisesFisherDistr::forMeanLength(Float l) {
	return (3*l - l*l*l) / (1-l*l);
}

