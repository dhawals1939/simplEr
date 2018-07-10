/*
 * scene.cpp
 *
 *  Created on: Nov 26, 2015
 *      Author: igkiou
 */

#include "scene.h"
#include "util.h"
#include <iostream>

namespace scn {

/*
 * Returns true if it is possible to intersect, and false otherwise.
 * disx is the smallest distance that must be traveled till intersection if outside the box. If inside the box, dis.x = 0;
 * disy is the smallest distance that must be traveled till intersection if inside the box.
 */
template <template <typename> class VectorType>
bool Block<VectorType>::intersect(const VectorType<Float> &p, const VectorType<Float> &d, Float &disx, Float &disy) const {

	VectorType<Float> l(M_MIN), r(M_MAX);
	for (int i = 0; i < p.dim; ++i) {
		if (std::abs(d[i]) > M_EPSILON * std::max(FPCONST(1.0), std::abs(d[i]))) {
			l[i] = (m_blockL[i] - p[i])/d[i];
			r[i] = (m_blockR[i] - p[i])/d[i];
			if (l[i] > r[i]) {
				std::swap(l[i], r[i]);
			}
		} else if (m_blockL[i] > p[i] || p[i] > m_blockR[i]) {
			return false;
		}
	}

	disx = l.max();
	disy = r.max();
	if (disx < FPCONST(0.0)) {
		disx = FPCONST(0.0);
	}
	if (disx - disy > -M_EPSILON) {
		return false;
	}

	return true;
}

template <template <typename> class VectorType>
bool Camera<VectorType>::samplePosition(VectorType<Float> &pos, smp::Sampler &sampler) const {
	pos = m_origin;
	for (int iter = 1; iter < m_origin.dim; ++iter) {
		pos[iter] += - m_plane[iter - 1] / FPCONST(2.0) + sampler() * m_plane[iter - 1];
	}
	return true;
}

template <template <typename> class VectorType>
bool AreaSource<VectorType>::sampleRay(VectorType<Float> &pos, VectorType<Float> &dir, smp::Sampler &sampler) const {
	pos = m_origin;
	for (int iter = 1; iter < m_origin.dim; ++iter) {
		pos[iter] += - m_plane[iter - 1] / FPCONST(2.0) + sampler() * m_plane[iter - 1];
	}
	dir = m_dir;
	return true;
}

template <template <typename> class VectorType>
bool Scene<VectorType>::genRay(VectorType<Float> &pos, VectorType<Float> &dir,
						smp::Sampler &sampler) const {

	if (m_source.sampleRay(pos, dir, sampler)) {
//		Float dist = FPCONST(0.0);
//		Assert(std::abs(dir.x) >= M_EPSILON);
//		if (dir.x >= M_EPSILON) {
//			dist = (m_mediumBlock.getBlockL().x - pos.x) / dir.x;
//		} else if (dir.x <= -M_EPSILON) {
//			dist = (m_mediumBlock.getBlockR().x - pos.x) / dir.x;
//		}
//		pos += dist * dir;
//		pos.x += M_EPSILON * 2;
		dir = m_refrDir;
		return true;
	} else {
		return false;
	}
}

template <template <typename> class VectorType>
bool Scene<VectorType>::genRay(VectorType<Float> &pos, VectorType<Float> &dir,
						smp::Sampler &sampler,
						VectorType<Float> &possrc, VectorType<Float> &dirsrc) const {

	if (m_source.sampleRay(pos, dir, sampler)) {
		possrc = pos;
		dirsrc = dir;
//		Float dist = FPCONST(0.0);
//		Assert(std::abs(dir.x) >= M_EPSILON);
//		if (dir.x >= M_EPSILON) {
//			dist = (m_mediumBlock.getBlockL().x - pos.x) / dir.x;
//		} else if (dir.x <= -M_EPSILON) {
//			dist = (m_mediumBlock.getBlockR().x - pos.x) / dir.x;
//		}
//		pos += dist * dir;
		dir = m_refrDir;
		return true;
	} else {
		return false;
	}
}

template <template <typename> class VectorType>
bool Scene<VectorType>::movePhoton(VectorType<Float> &p, VectorType<Float> &d,
									Float dist, smp::Sampler &sampler) const {

	VectorType<Float> p1 = p + dist * d;
	VectorType<Float> d1, norm;

	while (!m_block.inside(p1)) {
		Float disx, disy;
		if (!m_block.intersect(p, d, disx, disy)) {
			return false;
		}
		Assert(disx < M_EPSILON && dist - disy > -M_EPSILON);

		p += static_cast<Float>(disy)*d;
		dist -= static_cast<Float>(disy);

		int i;
		norm.zero();
		for (i = 0; i < p.dim; ++i) {
			if (std::abs(m_block.getBlockL()[i] - p[i]) < M_EPSILON) {
				norm[i] = -FPCONST(1.0);
				break;
			}
			else if (std::abs(m_block.getBlockR()[i] - p[i]) < M_EPSILON) {
				norm[i] = FPCONST(1.0);
				break;
			}
		}
		Assert(i < p.dim);

		Float minDiff = M_MAX;
		Float minDir = FPCONST(0.0);
		VectorType<Float> normalt;
		normalt.zero();
		int chosenI = p.dim;
		for (i = 0; i < p.dim; ++i) {
			Float diff = std::abs(m_block.getBlockL()[i] - p[i]);
			if (diff < minDiff) {
				minDiff = diff;
				chosenI = i;
				minDir = -FPCONST(1.0);
			}
			diff = std::abs(m_block.getBlockR()[i] - p[i]);
			if (diff < minDiff) {
				minDiff = diff;
				chosenI = i;
				minDir = FPCONST(1.0);
			}
		}
		normalt[chosenI] = minDir;
		Assert(normalt == norm);

		/*
		 * TODO: I think that, because we always return to same medium (we ignore
		 * refraction), there is no need to adjust radiance by eta*eta.
		 */
        m_bsdf.sample(d, norm, sampler, d1);
        if (tvec::dot(d1, norm) < FPCONST(0.0)) {
			// re-enter the medium through reflection
			d = d1;
		} else {
			return false;
		}
		p1 = p + dist*d;
	}

	p = p1;
	return true;
}

template <>
void Scene<tvec::TVector3>::addEnergyToImage(image::SmallImage &img, const tvec::Vec3f &p,
							Float pathlength, Float val) const {

	Float x = tvec::dot(m_camera.getHorizontal(), p) - m_camera.getOrigin().y;
	Float y = tvec::dot(m_camera.getVertical(), p) - m_camera.getOrigin().z;

	Assert(((std::abs(x) < FPCONST(0.5) * m_camera.getPlane().x)
				&& (std::abs(y) < FPCONST(0.5) * m_camera.getPlane().y)));
	if (((m_camera.getPathlengthRange().x == -1) && (m_camera.getPathlengthRange().y == -1)) ||
		((pathlength > m_camera.getPathlengthRange().x) && (pathlength < m_camera.getPathlengthRange().y))) {
		x = (x / m_camera.getPlane().x + FPCONST(0.5)) * static_cast<Float>(img.getXRes());
		y = (y / m_camera.getPlane().y + FPCONST(0.5)) * static_cast<Float>(img.getYRes());

//		int ix = static_cast<int>(img.getXRes()/2) + static_cast<int>(std::floor(x));
//		int iy = static_cast<int>(img.getYRes()/2) + static_cast<int>(std::floor(y));
		int ix = static_cast<int>(std::floor(x));
		int iy = static_cast<int>(std::floor(y));

		int iz;
		if ((m_camera.getPathlengthRange().x == -1) && (m_camera.getPathlengthRange().y == -1)) {
			iz = 0;
		} else {
			Float z = pathlength - m_camera.getPathlengthRange().x;
			Float range = m_camera.getPathlengthRange().y - m_camera.getPathlengthRange().x;
			z = (z / range) * static_cast<Float>(img.getZRes());
			iz = static_cast<int>(std::floor(z));
		}
#ifdef USE_PIXEL_SHARING
		Float fx = x - std::floor(x);
		Float fy = y - std::floor(y);

		addPixel(img, ix, iy, iz, val*(FPCONST(1.0) - fx)*(FPCONST(1.0) - fy));
		addPixel(img, ix + 1, iy, iz, val*fx*(FPCONST(1.0) - fy));
		addPixel(img, ix, iy + 1, iz, val*(FPCONST(1.0) - fx)*fy);
		addPixel(img, ix + 1, iy + 1, iz, val*fx*fy);
#else
		addPixel(img, ix, iy, iz, val);
#endif
    }
}

template <>
void Scene<tvec::TVector2>::addEnergyToImage(image::SmallImage &img, const tvec::Vec2f &p,
							Float pathlength, Float val) const {

	Float x = tvec::dot(m_camera.getHorizontal(), p) - m_camera.getOrigin().y;

	Assert(std::abs(x) < FPCONST(0.5) * m_camera.getPlane().x);
	if (((m_camera.getPathlengthRange().x == -1) && (m_camera.getPathlengthRange().y == -1)) ||
		((pathlength > m_camera.getPathlengthRange().x) && (pathlength < m_camera.getPathlengthRange().y))) {
		x = (x / m_camera.getPlane().x + FPCONST(0.5)) * static_cast<Float>(img.getXRes());

//		int ix = static_cast<int>(img.getXRes()/2) + static_cast<int>(std::floor(x));
//		int iy = static_cast<int>(img.getYRes()/2) + static_cast<int>(std::floor(y));
		int ix = static_cast<int>(std::floor(x));

		int iz;
		if ((m_camera.getPathlengthRange().x == -1) && (m_camera.getPathlengthRange().y == -1)) {
			iz = 0;
		} else {
			Float z = pathlength - m_camera.getPathlengthRange().x;
			Float range = m_camera.getPathlengthRange().y - m_camera.getPathlengthRange().x;
			z = (z / range) * static_cast<Float>(img.getZRes());
			iz = static_cast<int>(std::floor(z));
		}
#ifdef USE_PIXEL_SHARING
		Float fx = x - std::floor(x);

		addPixel(img, ix, 0, iz, val*(FPCONST(1.0) - fx));
		addPixel(img, ix + 1, 0, iz, val*fx);
#else
		addPixel(img, ix, 0, iz, val);
#endif
    }
}

template <template <typename> class VectorType>
void Scene<VectorType>::addEnergy(image::SmallImage &img,
			const VectorType<Float> &p, const VectorType<Float> &d, Float distTravelled,
			Float val, const med::Medium &medium, smp::Sampler &sampler) const {

#ifdef USE_WEIGHT_NORMALIZATION
	val *=	static_cast<Float>(img.getXRes()) * static_cast<Float>(img.getYRes())
		/ (m_camera.getPlane().x * m_camera.getPlane().y);
#ifdef USE_PRINTING
		std::cout << "using weight normalization " << std::endl;
#endif
#endif

	VectorType<Float> sensorPoint;
	if(m_camera.samplePosition(sensorPoint, sampler)) {
		/*
		 * TODO: Not sure why this check is here, but it was in the original code.
		 */
		Assert(m_block.inside(sensorPoint));

		VectorType<Float> distVec = sensorPoint - p;
		Float distToSensor = distVec.length();

		VectorType<Float> dirToSensor = distVec;
		dirToSensor.normalize();

		VectorType<Float> refrDirToSensor = dirToSensor;
		Float fresnelWeight = FPCONST(1.0);

		if (m_ior > FPCONST(1.0)) {
			Float sqrSum = FPCONST(0.0);
			for (int iter = 1; iter < dirToSensor.dim; ++iter) {
				refrDirToSensor[iter] = dirToSensor[iter] * m_ior;
				sqrSum += refrDirToSensor[iter] * refrDirToSensor[iter];
			}
			refrDirToSensor.x = std::sqrt(FPCONST(1.0) - sqrSum);
			if (dirToSensor.x < FPCONST(0.0)) {
				refrDirToSensor.x *= -FPCONST(1.0);
			}
#ifndef USE_NO_FRESNEL
			fresnelWeight = (FPCONST(1.0) -
			util::fresnelDielectric(dirToSensor.x, refrDirToSensor.x,
				FPCONST(1.0) / m_ior))
				/ m_ior / m_ior;
#endif
		}

		/*
		 * TODO: Double-check that the foreshortening term is needed, and
		 * that it is applied after refraction.
		 */
		Float foreshortening = dot(refrDirToSensor, m_camera.getDir());
		Assert(foreshortening >= FPCONST(0.0));
		Float totalDistance = (distTravelled + distToSensor) * m_ior;
		Float falloff = FPCONST(1.0);
		if (p.dim == 2) {
			falloff = distToSensor;
		} else if (p.dim == 3) {
			falloff = distToSensor * distToSensor;
		}
		Float totalPhotonValue = val
				* std::exp(- medium.getSigmaT() * distToSensor)
				* medium.getPhaseFunction()->f(d, dirToSensor)
				* fresnelWeight
				* foreshortening
				/ falloff;
		addEnergyToImage(img, sensorPoint, totalDistance, totalPhotonValue);
	}
}

template <template <typename> class VectorType>
void Scene<VectorType>::addEnergyDeriv(image::SmallImage &img, image::SmallImage &dSigmaT,
						image::SmallImage &dAlbedo, image::SmallImage &dGVal,
						const VectorType<Float> &p, const VectorType<Float> &d,
						Float distTravelled, Float val, Float sumScoreSigmaT,
						Float sumScoreAlbedo, Float sumScoreGVal,
						const med::Medium &medium, smp::Sampler &sampler) const {

#ifdef USE_WEIGHT_NORMALIZATION
	val *=	static_cast<Float>(img.getXRes()) * static_cast<Float>(img.getYRes())
		/ (m_camera.getPlane().x * m_camera.getPlane().y);
#ifdef USE_PRINTING
		std::cout << "using weight normalization " << std::endl;
#endif
#endif
#ifdef USE_PRINTING
	std::cout << "total = " << distTravelled << std::endl;
#endif

	VectorType<Float> sensorPoint;
	if(m_camera.samplePosition(sensorPoint, sampler)) {
		/*
		 * TODO: Not sure why this check is here, but it was in the original code.
		 */
		Assert(m_block.inside(sensorPoint));

		VectorType<Float> distVec = sensorPoint - p;
		Float distToSensor = distVec.length();

		VectorType<Float> dirToSensor = distVec;
		dirToSensor.normalize();

		VectorType<Float> refrDirToSensor = dirToSensor;
		Float fresnelWeight = FPCONST(1.0);

		if (m_ior > FPCONST(1.0)) {
			Float sqrSum = FPCONST(0.0);
			for (int iter = 1; iter < dirToSensor.dim; ++iter) {
				refrDirToSensor[iter] = dirToSensor[iter] * m_ior;
				sqrSum += refrDirToSensor[iter] * refrDirToSensor[iter];
			}
			refrDirToSensor.x = std::sqrt(FPCONST(1.0) - sqrSum);
			if (dirToSensor.x < FPCONST(0.0)) {
				refrDirToSensor.x *= -FPCONST(1.0);
			}
#ifndef USE_NO_FRESNEL
			fresnelWeight = (FPCONST(1.0) -
			util::fresnelDielectric(dirToSensor.x, refrDirToSensor.x,
				FPCONST(1.0) / m_ior))
				/ m_ior / m_ior;
#endif
		}

		/*
		 * TODO: Double-check that the foreshortening term is needed, and that
		 * it is applied after refraction.
		 */
		Float foreshortening = dot(refrDirToSensor, m_camera.getDir());
		Assert(foreshortening >= FPCONST(0.0));

		Float totalDistance = (distTravelled + distToSensor) * m_ior;
		Float falloff = FPCONST(1.0);
		if (p.dim == 2) {
			falloff = distToSensor;
		} else if (p.dim == 3) {
			falloff = distToSensor * distToSensor;
		}
		Float totalPhotonValue = val
				* std::exp(- medium.getSigmaT() * distToSensor)
				* medium.getPhaseFunction()->f(d, dirToSensor)
				* fresnelWeight
				* foreshortening
				/ falloff;
		addEnergyToImage(img, sensorPoint, totalDistance, totalPhotonValue);
		Float valDSigmaT = totalPhotonValue * (sumScoreSigmaT - distToSensor);
		addEnergyToImage(dSigmaT, sensorPoint, totalDistance, valDSigmaT);
		Float valDAlbedo = totalPhotonValue * sumScoreAlbedo;
		addEnergyToImage(dAlbedo, sensorPoint, totalDistance, valDAlbedo);
		Float valDGVal = totalPhotonValue *
				(sumScoreGVal + medium.getPhaseFunction()->score(d, dirToSensor));
		addEnergyToImage(dGVal, sensorPoint, totalDistance, valDGVal);
	}
}

template class Block<tvec::TVector2>;
template class Block<tvec::TVector3>;
template class Camera<tvec::TVector2>;
template class Camera<tvec::TVector3>;
template class AreaSource<tvec::TVector2>;
template class AreaSource<tvec::TVector3>;
template class Scene<tvec::TVector2>;
template class Scene<tvec::TVector3>;

}	/* namespace scn */
