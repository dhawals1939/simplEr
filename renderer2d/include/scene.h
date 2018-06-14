/*
 * scene.h
 *
 *  Created on: Nov 26, 2015
 *      Author: igkiou
 */

#ifndef SCENE_H_
#define SCENE_H_

#include <stdio.h>
#include <vector>

#include "constants.h"
#include "image.h"
#include "phase.h"
#include "sampler.h"
#include "tvector.h"
#include "medium.h"
#include "bsdf.h"
#include "util.h"

#include <omp.h>

namespace scn {

struct Block {

	Block(const tvec::Vec3f &blockL, const tvec::Vec3f &blockR)
		: m_blockL(blockL),
		  m_blockR(blockR) { }

	/*
	 * TODO: Maybe replace these with comparisons to FPCONST(0.0)?
	 */
	inline bool inside(const tvec::Vec3f &p) const {
		return
			p.x - m_blockL.x > -M_EPSILON &&
			m_blockR.x - p.x > -M_EPSILON &&
			p.y - m_blockL.y > -M_EPSILON &&
			m_blockR.y - p.y > -M_EPSILON &&
			p.z - m_blockL.z > -M_EPSILON &&
			m_blockR.z - p.z > -M_EPSILON;
//			p.x - m_boxL.x > M_EPSILON && m_boxR.x - p.x > M_EPSILON &&
//			p.y - m_blockL.y > M_EPSILON && m_blockR.y - p.y > M_EPSILON &&
//			p.z - m_blockL.z > M_EPSILON && m_blockR.z - p.z > M_EPSILON;
//			p.x - m_boxL.x > FPCONST(0.0) && m_boxR.x - p.x > FPCONST(0.0) &&
//			p.y - m_boxL.y > FPCONST(0.0) && m_boxR.y - p.y > FPCONST(0.0) &&
//			p.z - m_boxL.z > FPCONST(0.0) && m_boxR.z - p.z > FPCONST(0.0);
	}

	inline bool inside(const tvec::Vec3f &p, bool &xInside, bool &yInside, bool &zInside) const {
		xInside = p.x - m_blockL.x > -M_EPSILON &&
		m_blockR.x - p.x > -M_EPSILON;
		yInside = p.y - m_blockL.y > -M_EPSILON &&
		m_blockR.y - p.y > -M_EPSILON;
		zInside = p.z - m_blockL.z > -M_EPSILON &&
		m_blockR.z - p.z > -M_EPSILON;
//		xInside = p.x - m_boxL.x > M_EPSILON && m_boxR.x - p.x > M_EPSILON;
//		yInside = p.y - m_boxL.y > M_EPSILON && m_boxR.y - p.y > M_EPSILON;
//		zInside = p.z - m_boxL.z > M_EPSILON && m_boxR.z - p.z > M_EPSILON;
//		xInside = p.x - m_boxL.x > -M_EPSILON && m_boxR.x - p.x > -M_EPSILON;
//		yInside = p.y - m_boxL.y > -M_EPSILON && m_boxR.y - p.y > -M_EPSILON;
//		zInside = p.z - m_boxL.z > -M_EPSILON && m_boxR.z - p.z > -M_EPSILON;
		return xInside && yInside && zInside;
	}

//	bool intersect(const tvec::Vec3f &p, const tvec::Vec3f &d, tvec::Vec2d &dis) const;
	bool intersect(const tvec::Vec3f &p, const tvec::Vec3f &d,
				Float &disx, Float &disy) const;

	inline const tvec::Vec3f& getBlockL() const {
		return m_blockL;
	}

	inline const tvec::Vec3f& getBlockR() const {
		return m_blockR;
	}

	virtual ~Block() { }

protected:
	tvec::Vec3f m_blockL;
	tvec::Vec3f m_blockR;
};

struct Camera {
	/*
	 * TODO: Need to make sure m_x and m_y are used correctly. Maybe they are
	 * not needed at all..
	 */
	Camera(const tvec::Vec3f &origin, const tvec::Vec3f &dir,
		const tvec::Vec3f &x, const tvec::Vec3f &y,
		const tvec::Vec2f &plane, const tvec::Vec2f &pathlengthRange) :
			m_origin(origin),
			m_dir(dir),
			m_x(x),
			m_y(y),
			m_plane(plane),
			m_pathlengthRange(pathlengthRange) {
		Assert(((m_pathlengthRange.x == -FPCONST(1.0)) && (m_pathlengthRange.y == -FPCONST(1.0))) ||
				((m_pathlengthRange.x >= 0) && (m_pathlengthRange.y >= m_pathlengthRange.x)));
		/*
		 * TODO: Added this check for 2D version.
		 */
		Assert((m_origin.z == 0) && (m_dir.z == 0));
		m_dir.normalize();
		m_x.normalize();
		m_y.normalize();
	}

	/*
	 * TODO: Inline this method.
	 */
	bool samplePosition(tvec::Vec3f &pos, smp::Sampler &sampler) const;

	inline const tvec::Vec3f& getOrigin() const {
		return m_origin;
	}

	inline const tvec::Vec3f& getDir() const {
		return m_dir;
	}

	inline const tvec::Vec3f& getX() const {
		return m_x;
	}

	inline const tvec::Vec3f& getY() const {
		return m_y;
	}

	inline const tvec::Vec2f& getPlane() const {
		return m_plane;
	}

	inline const tvec::Vec2f& getPathlengthRange() const {
		return m_pathlengthRange;
	}

	virtual ~Camera() { }

protected:
	tvec::Vec3f m_origin;
	tvec::Vec3f m_dir;
	tvec::Vec3f m_x;
	tvec::Vec3f m_y;
	tvec::Vec2f m_plane;
	tvec::Vec2f m_pathlengthRange;
};

struct AreaSource {

	AreaSource(const tvec::Vec3f &origin, const tvec::Vec3f &dir,
			const tvec::Vec2f &plane, Float Li)
			: m_origin(origin),
			  m_dir(dir),
			  m_plane(plane),
			  m_Li(Li) { /* m_dir(std::cos(angle), std::sin(angle), FPCONST(0.0)) */
		/*
		 * TODO: Added this check for 2D version.
		 */
		Assert((m_origin.z == 0) && (m_dir.z == 0));
		m_dir.normalize();
#ifdef USE_PRINTING
		std::cout << " dir " << m_dir.x << " " << m_dir.y << " " << m_dir.z << std::endl;
#endif
	}

	/*
	 * TODO: Inline this method.
	 */
	bool sampleRay(tvec::Vec3f &pos, tvec::Vec3f &dir, smp::Sampler &sampler) const;

	inline const tvec::Vec3f& getOrigin() const {
		return m_origin;
	}

	inline const tvec::Vec3f& getDir() const {
		return m_dir;
	}

	inline const tvec::Vec2f& getPlane() const {
		return m_plane;
	}

	inline Float getLi() const {
		return m_Li;
	}

	virtual ~AreaSource() { }

protected:
	tvec::Vec3f m_origin;
	tvec::Vec3f m_dir;
	tvec::Vec2f m_plane;
	Float m_Li;
};

class Scene {
public:
	Scene(const Float ior,
			const tvec::Vec3f &blockL,
			const tvec::Vec3f &blockR,
			const tvec::Vec3f &lightOrigin,
			const tvec::Vec3f &lightDir,
			const tvec::Vec2f &lightPlane,
			const Float Li,
			const tvec::Vec3f &viewOrigin,
			const tvec::Vec3f &viewDir,
			const tvec::Vec3f &viewX,
			const tvec::Vec3f &viewY,
			const tvec::Vec2f &viewPlane,
			const tvec::Vec2f &pathlengthRange) :
				m_ior(ior),
				m_fresnelTrans(FPCONST(1.0)),
				m_refrDir(FPCONST(1.0), FPCONST(0.0), FPCONST(0.0)),
				m_block(blockL, blockR),
				m_source(lightOrigin, lightDir, lightPlane, Li),
				m_camera(viewOrigin, viewDir, viewX, viewY, viewPlane, pathlengthRange),
				m_bsdf(FPCONST(1.0), ior) {

		Assert(((std::abs(m_source.getOrigin().x - m_block.getBlockL().x) < M_EPSILON) && (m_source.getDir().x > FPCONST(0.0)))||
				((std::abs(m_source.getOrigin().x - m_block.getBlockR().x) < M_EPSILON) && (m_source.getDir().x < FPCONST(0.0))));

		if (m_ior > FPCONST(1.0)) {
			m_refrDir.y = m_source.getDir().y / m_ior;
			m_refrDir.z = m_source.getDir().z / m_ior;
			m_refrDir.x = std::sqrt(FPCONST(1.0) - m_refrDir.y * m_refrDir.y - m_refrDir.z * m_refrDir.z);
			if (m_source.getDir().x < FPCONST(0.0)) {
				m_refrDir.x *= -FPCONST(1.0);
			}
#ifndef USE_NO_FRESNEL
			m_fresnelTrans = m_ior * m_ior
					* (FPCONST(1.0) -
					util::fresnelDielectric(m_source.getDir().x, m_refrDir.x, m_ior));
#endif
		} else {
			m_refrDir = m_source.getDir();
		}
#ifdef USE_PRINTING
		std::cout << "fresnel " << m_fresnelTrans << std::endl;
#endif
	}

	/*
	 * TODO: Inline these methods in implementations.
	 */
	bool movePhoton(tvec::Vec3f &p, tvec::Vec3f &d, Float dist,
					smp::Sampler &sampler) const;
	bool genRay(tvec::Vec3f &pos, tvec::Vec3f &dir, smp::Sampler &sampler) const;
	bool genRay(tvec::Vec3f &pos, tvec::Vec3f &dir, smp::Sampler &sampler,
				tvec::Vec3f &possrc, tvec::Vec3f &dirsrc) const;
	void addEnergyToImage(image::SmallImage &img, const tvec::Vec3f &p,
						Float pathlength, Float val) const;

	inline void addPixel(image::SmallImage &img, int x, int y, int z, Float val) const {
		if (x >= 0 && x < img.getXRes() && y >= 0 && y < img.getYRes() &&
			z >= 0 && z < img.getZRes()) {
			img.addEnergy(x, y, z, static_cast<Float>(val));
		}
	}

	void addEnergy(image::SmallImage &img, const tvec::Vec3f &p,
						const tvec::Vec3f &d, Float distTravelled, Float val,
						const med::Medium &medium, smp::Sampler &sampler) const;

	void addEnergyDeriv(image::SmallImage &img, image::SmallImage &dSigmaT,
						image::SmallImage &dAlbedo, image::SmallImage &dGVal,
						const tvec::Vec3f &p, const tvec::Vec3f &d,
						Float distTravelled, Float val, Float sumScoreSigmaT,
						Float sumScoreAlbedo, Float sumScoreGVal,
						const med::Medium &medium, smp::Sampler &sampler) const;

	/*
	 * TODO: Direct lighting is currently not supported.
	 */
//	void addEnergyDirect(image::SmallImage &img, const tvec::Vec3f &p,
//						const tvec::Vec3f &d, Float val,
//						const med::Medium &medium, smp::Sampler &sampler) const;

	inline Float getMediumIor() const {
		return m_ior;
	}

	inline Float getFresnelTrans() const {
		return m_fresnelTrans;
	}

	inline const tvec::Vec3f& getRefrDir() const {
		return m_refrDir;
	}

	inline const Block& getMediumBlock() const {
		return m_block;
	}

	inline const AreaSource& getAreaSource() const {
		return m_source;
	}

	inline const Camera& getCamera() const {
		return m_camera;
	}

	inline const bsdf::SmoothDielectric& getBSDF() const {
		return m_bsdf;
	}

	~Scene() { }

protected:
	Float m_ior;
	Float m_fresnelTrans;
	tvec::Vec3f m_refrDir;
	Block m_block;
	AreaSource m_source;
	Camera m_camera;
	bsdf::SmoothDielectric m_bsdf;
};

}	/* namespace scn */

#endif /* SCENE_H_ */
