/*
 * util.h
 *
 *  Created on: Nov 15, 2015
 *      Author: igkiou
 */
#pragma once
#ifndef UTIL_H_
#define UTIL_H_

#include "tvector.h"
#include "constants.h"
#include <vector>
#include <iterator>
#include <algorithm>

namespace util {

template <typename T>
inline void coordinateSystem(const tvec::TVector3<T> &a, tvec::TVector3<T> &b, tvec::TVector3<T> &c) {
	if (std::abs(a.x) > std::abs(a.y)) {
		Float invLen = FPCONST(1.0) / std::sqrt(a.x * a.x + a.z *a.z);
		c = tvec::TVector3<T>(a.z * invLen, FPCONST(0.0), -a.x * invLen);
	} else {
		Float invLen = FPCONST(1.0) / std::sqrt(a.y * a.y + a.z * a.z);
		c = tvec::TVector3<T>(FPCONST(0.0), a.z * invLen, -a.y * invLen);
	}
	b = tvec::cross(c, a);
}

template <typename T>
inline void coordinateSystem(const tvec::TVector2<T> &a, tvec::TVector2<T> &b) {
	/*
	 * TODO: Changed this for 2D version.
	 */
	b = tvec::TVector2<T>(a.y, -a.x);
}

/*
 * TODO: Need to make sure that, every time this is called, the local code also
 * takes care to apply the eta*eta scaling if needed. Note that, according to
 * mitsuba, this scaling is needed when we return to the same medium through TIR.
 */
template <typename T>
inline Float fresnelDielectric(T cosThetaI, T cosThetaT, T eta)
{
	if (std::abs(eta - FPCONST(1.0)) < M_EPSILON * std::max(FPCONST(1.0), std::abs(eta))) {
		return FPCONST(0.0);
	} else {
		T Rs = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
		T Rp = (cosThetaT - eta * cosThetaI) / (cosThetaT + eta * cosThetaI);

		/* No polarization -- return the unpolarized reflectance */
		/*
		 * TODO: May need to fix for polarization.
		 */
		return FPCONST(0.5) * static_cast<Float>(Rs * Rs + Rp * Rp);
	}
}

template <typename T>
inline T safeSqrt(T x) {
    return x > static_cast<T>(0.0) ? std::sqrt(x) : static_cast<T>(0.0);
}

template <typename T, template <typename> class VectorType>
inline void reflect(const VectorType<T> &a,
					const VectorType<T> &n,
					VectorType<T> &b) {
    b = -static_cast<T>(2.0)*tvec::dot(a, n)*n + a;
}

template <typename T, template <typename> class VectorType>
inline bool refract(const VectorType<T> &a,
					const VectorType<T> &n,
					T eta,
					VectorType<T> &b) {

	VectorType<T> q = tvec::dot(a, n)*n;
	VectorType<T> p = (a - q)/eta;
    if ( p.length() > static_cast<T>(1.0) ) {
        reflect(a, n, b);
        return false;
    } else {
        q.normalize();
        q *= util::safeSqrt(static_cast<T>(1.0) - p.lengthSquared());
        b = p + q;
        return true;
   }
}

template <typename T, template <typename> class VectorType>
inline bool rayIntersectBox(const VectorType<T> &p, const VectorType<T> &d,
                             const VectorType<T> &min, const VectorType<T> &max,
							 T &t1, T &t2) {
	t1 = M_MIN;
	t2 = M_MAX;
	for (int i = 0; i < p.dim; ++i) {
		if (std::abs(d[i]) > M_EPSILON) {
			T v1, v2;
			if ( d[i] > static_cast<T>(0.0) ) {
				v1 = (min[i] - p[i])/d[i];
				v2 = (max[i] - p[i])/d[i];
			} else {
				v1 = (max[i] - p[i])/d[i];
				v2 = (min[i] - p[i])/d[i];
			}

			if (v1 > t1) {
				t1 = v1;
			}
			if (v2 < t2) {
				t2 = v2;
			}
		} else if (min[i] - p[i] > M_EPSILON || p[i] - max[i] > M_EPSILON) {
			return false;
		}
	}
    return t2 > t1;
}


}	/* namespace util */

#endif /* UTIL_H_ */
