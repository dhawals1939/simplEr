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
bool Block::intersect(const tvec::Vec3f &p, const tvec::Vec3f &d, Float &disx, Float &disy) const {
	tvec::Vec3d l(M_MIN), r(M_MAX);

	for (int i = 0; i < 3; ++i) {
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

	disx = l[0];
	if (l[1] > disx) {
		disx = l[1];
	}
	if (l[2] > disx) {
		disx = l[2];
	}
	disy = r[0];
	if (disy > r[1]) {
		disy = r[1];
	}
	if (disy > r[2]) {
		disy = r[2];
	}
	if (disx < FPCONST(0.0)) {
		disx = FPCONST(0.0);
	}
	if (disx - disy > -M_EPSILON) {
		return false;
	}

	return true;
}

bool Camera::samplePosition(tvec::Vec3f &pos, smp::Sampler &sampler) const {
	Float xi1 = sampler();
	Float xi2 = sampler();
	Float yCoordShift = - m_plane.x / FPCONST(2.0) + xi1 * m_plane.x;
	Float zCoordShift = - m_plane.y / FPCONST(2.0) + xi2 * m_plane.y;

	pos = m_origin + tvec::Vec3f(FPCONST(0.0), yCoordShift, zCoordShift);
	return true;
}

bool AreaSource::sampleRay(tvec::Vec3f &pos, tvec::Vec3f &dir, smp::Sampler &sampler) const {
	Float xi1 = sampler();
	Float xi2 = sampler();
	Float yCoordShift = - m_plane.x / FPCONST(2.0) + xi1 * m_plane.x;
	Float zCoordShift = - m_plane.y / FPCONST(2.0) + xi2 * m_plane.y;

	pos = m_origin + tvec::Vec3f(FPCONST(0.0), yCoordShift, zCoordShift);
	dir = m_dir;
	return true;
}

bool Scene::genRay(tvec::Vec3f &pos, tvec::Vec3f &dir,
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

bool Scene::genRay(tvec::Vec3f &pos, tvec::Vec3f &dir,
						smp::Sampler &sampler,
						tvec::Vec3f &possrc, tvec::Vec3f &dirsrc) const {

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

bool Scene::movePhoton(tvec::Vec3f &p, tvec::Vec3f &d, Float dist,
					smp::Sampler &sampler) const {

	tvec::Vec3f p1 = p + dist * d;
    tvec::Vec3f d1, norm;

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
		for (i = 0; i < 3; ++i) {
			if (std::abs(m_block.getBlockL()[i] - p[i]) < M_EPSILON) {
				norm[i] = -FPCONST(1.0);
				break;
			}
			else if (std::abs(m_block.getBlockR()[i] - p[i]) < M_EPSILON) {
				norm[i] = FPCONST(1.0);
				break;
			}
		}
		Assert(i < 3);

		Float minDiff = M_MAX;
		Float minDir = FPCONST(0.0);
		tvec::Vec3f normalt;
		normalt.zero();
		int chosenI = 3;
		for (i = 0; i < 3; ++i) {
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

void Scene::addEnergyToImage(image::SmallImage &img, const tvec::Vec3f &p,
							Float pathlength, Float val) const {
	/*
	 * TODO: I don't think we need viewX and viewY anymore.
	 */
	Float x = tvec::dot(m_camera.getX(), p) - m_camera.getOrigin().x;
	Float y = tvec::dot(m_camera.getY(), p) - m_camera.getOrigin().y;


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

void Scene::addEnergy(image::SmallImage &img,
			const tvec::Vec3f &p, const tvec::Vec3f &d, Float distTravelled,
			Float val, const med::Medium &medium, smp::Sampler &sampler) const {

	tvec::Vec3f sensorPoint, dirToSensor, refrDirToSensor;
	Float distToSensor;
	Float foreshortening;
	Float fresnelWeight(1.0);
	Float totalPhotonValue;
	Float totalDistance;

#ifdef USE_WEIGHT_NORMALIZATION
	val *=	static_cast<Float>(img.getXRes()) * static_cast<Float>(img.getYRes())
		/ (m_camera.getPlane().x * m_camera.getPlane().y);
#ifdef USE_PRINTING
		std::cout << "using weight normalization " << std::endl;
#endif
#endif

	if(m_camera.samplePosition(sensorPoint, sampler)) {

		tvec::Vec3f distVec = sensorPoint - p;
		distToSensor = distVec.length();
		dirToSensor = distVec;
		dirToSensor.normalize();
		/*
		 * TODO: Not sure why this check is here, but it was in the original code.
		 */
		if (m_block.inside(sensorPoint)) {

			if (m_ior > FPCONST(1.0)) {
				refrDirToSensor.y = dirToSensor.y * m_ior;
				refrDirToSensor.z = dirToSensor.z * m_ior;
				refrDirToSensor.x = std::sqrt(FPCONST(1.0)
									- dirToSensor.y * dirToSensor.y
									- dirToSensor.z * dirToSensor.z);
				if (dirToSensor.x < FPCONST(0.0)) {
					dirToSensor.x *= -FPCONST(1.0);
				}
#ifndef USE_NO_FRESNEL
				fresnelWeight = (FPCONST(1.0) -
				util::fresnelDielectric(dirToSensor.x, refrDirToSensor.x,
					FPCONST(1.0) / m_ior))
					/ m_ior / m_ior;
#endif
			} else {
				refrDirToSensor = dirToSensor;
			}

			/*
			 * TODO: Double-check that the foreshortening term is needed, and
			 * that it is applied after refraction.
			 */
			foreshortening = dot(refrDirToSensor, m_camera.getDir());
			Assert(foreshortening <= FPCONST(0.0));
			totalDistance = (distTravelled + distToSensor) * m_ior;
			totalPhotonValue = val * std::exp(- medium.getSigmaT() * distToSensor)
					* medium.getPhaseFunction()->f(d, dirToSensor) * fresnelWeight;
			addEnergyToImage(img, sensorPoint, totalDistance, totalPhotonValue);
		} else {
			Assert(m_block.inside(sensorPoint));
		}
	}
}

void Scene::addEnergyDeriv(image::SmallImage &img, image::SmallImage &dSigmaT,
						image::SmallImage &dAlbedo, image::SmallImage &dGVal,
						const tvec::Vec3f &p, const tvec::Vec3f &d,
						Float distTravelled, Float val, Float sumScoreSigmaT,
						Float sumScoreAlbedo, Float sumScoreGVal,
						const med::Medium &medium, smp::Sampler &sampler) const {

	tvec::Vec3f sensorPoint, dirToSensor, refrDirToSensor;
	Float distToSensor;
	Float foreshortening;
	Float fresnelWeight(1.0);
	Float totalPhotonValue, valDSigmaT, valDAlbedo, valDGVal;
	Float totalDistance;

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

	if(m_camera.samplePosition(sensorPoint, sampler)) {

		tvec::Vec3f distVec = sensorPoint - p;
		distToSensor = distVec.length();
		dirToSensor = distVec;
		dirToSensor.normalize();
		/*
		 * TODO: Not sure why this check is here, but it was in the original code.
		 */
		if (m_block.inside(sensorPoint)) {

			if (m_ior > FPCONST(1.0)) {
				refrDirToSensor.y = dirToSensor.y * m_ior;
				refrDirToSensor.z = dirToSensor.z * m_ior;
				refrDirToSensor.x = std::sqrt(FPCONST(1.0)
									- dirToSensor.y * dirToSensor.y
									- dirToSensor.z * dirToSensor.z);
				if (dirToSensor.x < FPCONST(0.0)) {
					dirToSensor.x *= -FPCONST(1.0);
				}
#ifndef USE_NO_FRESNEL
				fresnelWeight = (FPCONST(1.0) -
				util::fresnelDielectric(dirToSensor.x, refrDirToSensor.x,
					FPCONST(1.0) / m_ior))
					/ m_ior / m_ior;
#endif
			} else {
				refrDirToSensor = dirToSensor;
			}

			/*
			 * TODO: Double-check that the foreshortening term is needed, and that
			 * it is applied after refraction.
			 * TODO: Here I'll also need to include a Fresnel transmission term.
			 */
			foreshortening = dot(refrDirToSensor, m_camera.getDir());
			Assert(foreshortening <= FPCONST(0.0));

			totalDistance = (distTravelled + distToSensor) * m_ior;
			totalPhotonValue = val * std::exp(- medium.getSigmaT() * distToSensor)
					* medium.getPhaseFunction()->f(d, dirToSensor) * fresnelWeight;
			addEnergyToImage(img, sensorPoint, totalDistance, totalPhotonValue);

			valDSigmaT = totalPhotonValue * (sumScoreSigmaT - distToSensor);
			addEnergyToImage(dSigmaT, sensorPoint, totalDistance, valDSigmaT);
			valDAlbedo = totalPhotonValue * sumScoreAlbedo;
			addEnergyToImage(dAlbedo, sensorPoint, totalDistance, valDAlbedo);
			valDGVal = totalPhotonValue *
					(sumScoreGVal + medium.getPhaseFunction()->score(d, dirToSensor));
			addEnergyToImage(dGVal, sensorPoint, totalDistance, valDGVal);

		} else {
			Assert(m_block.inside(sensorPoint));
		}
	}
}

//void Scene::addEnergyDirect(image::SmallImage &img,
//			const tvec::Vec3f &p, const tvec::Vec3f &d, Float val,
//			const med::Medium &medium) const {
//
//	tvec::Vec3f q;
//	Float t;
//
//#ifdef USE_WEIGHT_NORMALIZATION
//	val *=	static_cast<Float>(img.getXRes()) * static_cast<Float>(img.getYRes())
//		/ (m_camera.getViewPlane().x * m_camera.getViewPlane().y);
//#ifdef USE_PRINTING
//		std::cout << "using weight normalization " << std::endl;
//#endif
//#endif
//
//	if (std::abs(m_refX.x) > M_EPSILON) {
//		t = ((m_refX.x > FPCONST(0.0) ? m_mediumBlock.getBlockR().x : m_mediumBlock.getBlockL().x) - p.x)/m_refX.x;
//		q = p + t*m_refX;
//		if (m_mediumBlock.inside(q)) {
//			if (d.aproxEqual(m_refX)) {
//				addEnergyToImage(img, q, 0, val*std::exp(-medium.getSigmaT()*t));
//			}
//		}
//	}
//
//	if (std::abs(m_refY.y) > M_EPSILON) {
//		t = ((m_refY.y > FPCONST(0.0) ? m_mediumBlock.getBlockR().y : m_mediumBlock.getBlockL().y) - p.y)/m_refY.y;
//		q = p + t*m_refY;
//		if (m_mediumBlock.inside(q)) {
//			if (d.aproxEqual(m_refY)) {
//				addEnergyToImage(img, q, 0, val*std::exp(-medium.getSigmaT()*t));
//			}
//		}
//	}
//
//	if (std::abs(m_refZ.z) > M_EPSILON) {
//		t = ((m_refZ.z > FPCONST(0.0) ? m_mediumBlock.getBlockR().z : m_mediumBlock.getBlockL().z) - p.z)/m_refZ.z;
//		q = p + t*m_refZ;
//		if (m_mediumBlock.inside(q)) {
//			if (d.aproxEqual(m_refZ)) {
//				addEnergyToImage(img, q, 0, val*std::exp(-medium.getSigmaT()*t));
//			}
//		}
//	}
//}

}	/* namespace scn */
