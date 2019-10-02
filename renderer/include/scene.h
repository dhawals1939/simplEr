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
#include "pmf.h"
#include "sampler.h"
#include "tvector.h"
#include "medium.h"
#include "bsdf.h"
#include "util.h"

#include <omp.h>

namespace scn {

template <template <typename> class VectorType>
struct Block {

	Block(const VectorType<Float> &blockL, const VectorType<Float> &blockR)
		: m_blockL(blockL),
		  m_blockR(blockR) { }

	/*
	 * TODO: Maybe replace these with comparisons to FPCONST(0.0)?
	 */
	inline bool inside(const VectorType<Float> &p) const {
		bool result = true;
		for (int iter = 0; iter < p.dim; ++iter) {
			result = result
				&& (p[iter] - m_blockL[iter] > -M_EPSILON)
				&& (m_blockR[iter] - p[iter] > -M_EPSILON);
			/*
			 * TODO: Maybe remove this check, it may be slowing performance down
			 * due to the branching.
			 */
			if (!result) {
				break;
			}
		}
		return result;
	}

	bool intersect(const VectorType<Float> &p, const VectorType<Float> &d,
				Float &disx, Float &disy) const;


	inline const VectorType<Float>& getBlockL() const {
		return m_blockL;
	}

	inline const VectorType<Float>& getBlockR() const {
		return m_blockR;
	}

	virtual ~Block() { }

protected:
	VectorType<Float> m_blockL;
	VectorType<Float> m_blockR;
};

template <template <typename> class VectorType>
struct Camera {

	Camera(const VectorType<Float> &origin,
		const VectorType<Float> &dir,
		const VectorType<Float> &horizontal,
		const tvec::Vec2f &plane,
		const tvec::Vec2f &pathlengthRange) :
			m_origin(origin),
			m_dir(dir),
			m_horizontal(horizontal),
			m_vertical(),
			m_plane(plane),
			m_pathlengthRange(pathlengthRange) {
		Assert(((m_pathlengthRange.x == -FPCONST(1.0)) && (m_pathlengthRange.y == -FPCONST(1.0))) ||
				((m_pathlengthRange.x >= 0) && (m_pathlengthRange.y >= m_pathlengthRange.x)));
		m_dir.normalize();
		m_horizontal.normalize();
		if (m_origin.dim == 3) {
			m_vertical = tvec::cross(m_dir, m_horizontal);
		}
	}

	/*
	 * TODO: Inline this method.
	 */
	bool samplePosition(VectorType<Float> &pos, smp::Sampler &sampler) const;

	inline const VectorType<Float>& getOrigin() const {
		return m_origin;
	}

	inline const VectorType<Float>& getDir() const {
		return m_dir;
	}

	inline const VectorType<Float>& getHorizontal() const {
		return m_horizontal;
	}

	inline const VectorType<Float>& getVertical() const {
		return m_vertical;
	}

	inline const tvec::Vec2f& getPlane() const {
		return m_plane;
	}

	inline const tvec::Vec2f& getPathlengthRange() const {
		return m_pathlengthRange;
	}

	virtual ~Camera() { }

protected:
	VectorType<Float> m_origin;
	VectorType<Float> m_dir;
	VectorType<Float> m_horizontal;
	VectorType<Float> m_vertical;
	tvec::Vec2f m_plane;
	tvec::Vec2f m_pathlengthRange;
};

template <template <typename> class VectorType>
struct AreaTexturedSource {


	enum EmitterType{directional, diffuse}; //diffuse is still not implemented

	AreaTexturedSource(const VectorType<Float> &origin, const VectorType<Float> &dir, const std::string& filename,
			const tvec::Vec2f &plane, Float Li, const EmitterType &emittertype = EmitterType::directional)
			: m_origin(origin),
			  m_dir(dir),
			  m_emittertype(emittertype),
			  m_plane(plane),
			  m_Li(Li) { /* m_dir(std::cos(angle), std::sin(angle), FPCONST(0.0)) */
		m_texture.readFile(filename);
		int _length = m_texture.getXRes()*m_texture.getYRes();
		m_pixelsize.x = m_plane.x/m_texture.getXRes();
		m_pixelsize.y = m_plane.y/m_texture.getYRes();


		m_textureSampler.reserve(_length);
		for(int i=0; i<_length; i++){
			m_textureSampler.append(m_texture.getPixel(i));
		}
		m_textureSampler.normalize();
	}

	bool sampleRay(VectorType<Float> &pos, VectorType<Float> &dir, smp::Sampler &sampler) const;

	inline const VectorType<Float>& getOrigin() const {
		return m_origin;
	}

	inline const VectorType<Float>& getDir() const {
		return m_dir;
	}

	inline const tvec::Vec2f& getPlane() const {
		return m_plane;
	}

	inline Float getLi() const {
		return m_Li;
	}

	virtual ~AreaTexturedSource() { }

protected:
	VectorType<Float> m_origin;
	VectorType<Float> m_dir;
	image::Texture m_texture;
	DiscreteDistribution m_textureSampler;
	tvec::Vec2f m_pixelsize;
	tvec::Vec2f m_plane;
	Float m_Li;
	EmitterType m_emittertype;
};


template <template <typename> class VectorType>
struct AreaSource {

	AreaSource(const VectorType<Float> &origin, const VectorType<Float> &dir,
			const tvec::Vec2f &plane, Float Li)
			: m_origin(origin),
			  m_dir(dir),
			  m_plane(plane),
			  m_Li(Li) { /* m_dir(std::cos(angle), std::sin(angle), FPCONST(0.0)) */
		/*
		 * TODO: Added this check for 2D version.
		 */
		m_dir.normalize();
#ifdef USE_PRINTING
		std::cout << " dir " << m_dir.x << " " << m_dir.y;
		if (m_dir.dim == 3) {
			std::cout << " " << m_dir.z << std::endl;
		} else {
			std::cout << std::endl;
		}
#endif
	}

	/*
	 * TODO: Inline this method.
	 */
	bool sampleRay(VectorType<Float> &pos, VectorType<Float> &dir, smp::Sampler &sampler) const;

	inline const VectorType<Float>& getOrigin() const {
		return m_origin;
	}

	inline const VectorType<Float>& getDir() const {
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
	VectorType<Float> m_origin;
	VectorType<Float> m_dir;
	tvec::Vec2f m_plane;
	Float m_Li;
};


template <template <typename> class VectorType>
struct US {
	Float    f_u;          // Ultrasound frequency (1/s or Hz)
	Float    speed_u;      // Ultrasound speed (m/s)
	Float  wavelength_u; // (m)

	Float n_o;          // Baseline refractive index
	Float n_max;        // Max refractive index variation
	Float k_r;
	Float z_max;        // Depth of the scattering medium (in m)
    int    mode;         // Order of the bessel function or mode of the ultrasound

    VectorType<Float>    axis_uz;          // Ultrasound axis
    VectorType<Float>    axis_ux;          // Ultrasound x-axis. Need to compute angle as mode > 0 is a function of phi

    VectorType<Float>    p_u;             // A point on the ultra sound axis

    Float er_stepsize;


    US(const Float& f_u, const Float& speed_u,
                 const Float& n_o, const Float& n_max, const int& mode,
                 const VectorType<Float> &axis_uz, const VectorType<Float> &axis_ux, const VectorType<Float> &p_u, const Float &er_stepsize){
        this->f_u            = f_u;
		this->speed_u        = speed_u;      
		this->wavelength_u   = ((double) speed_u)/f_u; 
		this->n_o            = n_o;         
		this->n_max          = n_max;      
		this->k_r            = (2*M_PI)/wavelength_u;
		this->mode           = mode;     

		this->axis_uz        = axis_uz;
		this->axis_ux        = axis_ux;
		this->p_u            = p_u;

		this->er_stepsize    = er_stepsize;
    }

    double RIF(const VectorType<Float> &p) const;

    const VectorType<Float> dRIF(const VectorType<Float> &q) const;

};


template <template <typename> class VectorType>
class Scene {
public:
	Scene(Float ior,
			const VectorType<Float> &blockL,
			const VectorType<Float> &blockR,
			const VectorType<Float> &lightOrigin,
			const VectorType<Float> &lightDir,
			const std::string &lightTextureFile,
			const tvec::Vec2f &lightPlane,
			const Float Li,
			const VectorType<Float> &viewOrigin,
			const VectorType<Float> &viewDir,
			const VectorType<Float> &viewHorizontal,
			const tvec::Vec2f &viewPlane,
			const tvec::Vec2f &pathlengthRange, 
			//Ultrasound parameters: a lot of them are currently not used
			const Float& f_u,
			const Float& speed_u,
			const Float& n_o,
			const Float& n_max,
			const int& mode,
			const VectorType<Float> &axis_uz,
			const VectorType<Float> &axis_ux,
			const VectorType<Float> &p_u,
			const Float &er_stepsize
            ) :
				m_ior(ior),
				m_fresnelTrans(FPCONST(1.0)),
				m_refrDir(),
				m_block(blockL, blockR),
				m_source(lightOrigin, lightDir, lightPlane, Li),
//				m_source(lightOrigin, lightDir, lightTextureFile, lightPlane, Li),
				m_camera(viewOrigin, viewDir, viewHorizontal, viewPlane, pathlengthRange),
				m_bsdf(FPCONST(1.0), ior),
				m_us(f_u, speed_u, n_o, n_max, mode, axis_uz, axis_ux, p_u, er_stepsize){

		Assert(((std::abs(m_source.getOrigin().x - m_block.getBlockL().x) < M_EPSILON) && (m_source.getDir().x > FPCONST(0.0)))||
				((std::abs(m_source.getOrigin().x - m_block.getBlockR().x) < M_EPSILON) && (m_source.getDir().x < FPCONST(0.0))));

		if (m_ior > FPCONST(1.0)) {
			Float sumSqr = FPCONST(0.0);
			for (int iter = 1; iter < m_refrDir.dim; ++iter) {
				m_refrDir[iter] = m_source.getDir()[iter] / m_ior;
				sumSqr += m_refrDir[iter] * m_refrDir[iter];
			}
			m_refrDir.x = std::sqrt(FPCONST(1.0) - sumSqr);
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
	 * ER trackers
	 */
	inline VectorType<Float> dP(const VectorType<Float> d) const{
		return d;
	}
	inline VectorType<Float> dOmega(const VectorType<Float> p, const VectorType<Float> d) const{
		VectorType<Float> dn = m_us.dRIF(p);
		Float              n = m_us.RIF(p);

		return (dn - dot(d, dn)*d)/n;
	}

	void er_step(VectorType<Float> &p, VectorType<Float> &d, const Float &stepSize) const;
	void trace(VectorType<Float> &p, VectorType<Float> &d, const Float &distance) const; // Non optical
	void traceTillBlock(VectorType<Float> &p, VectorType<Float> &d, const Float &dist, Float &disx, Float &disy) const;
	void trace_optical_distance(VectorType<Float> &p, VectorType<Float> &d, const Float &distance) const; // optical

	/*
	 * TODO: Inline these methods in implementations.
	 */
	bool movePhotonTillSensor(VectorType<Float> &p, VectorType<Float> &d, Float &distToSensor,
					smp::Sampler &sampler) const;
	bool movePhoton(VectorType<Float> &p, VectorType<Float> &d, Float dist,
					smp::Sampler &sampler) const;
	bool genRay(VectorType<Float> &pos, VectorType<Float> &dir, smp::Sampler &sampler) const;
	bool genRay(VectorType<Float> &pos, VectorType<Float> &dir, smp::Sampler &sampler,
				VectorType<Float> &possrc, VectorType<Float> &dirsrc) const;
	void addEnergyToImage(image::SmallImage &img, const VectorType<Float> &p,
						Float pathlength, Float val) const;

	inline void addPixel(image::SmallImage &img, int x, int y, int z, Float val) const {
		if (x >= 0 && x < img.getXRes() && y >= 0 && y < img.getYRes() &&
			z >= 0 && z < img.getZRes()) {
			img.addEnergy(x, y, z, static_cast<Float>(val));
		}
	}

	void addEnergyInParticle(image::SmallImage &img, const VectorType<Float> &p,
						const VectorType<Float> &d, Float distTravelled, Float val,
						const med::Medium &medium, smp::Sampler &sampler) const;

	void addEnergy(image::SmallImage &img, const VectorType<Float> &p,
						const VectorType<Float> &d, Float distTravelled, Float val,
						const med::Medium &medium, smp::Sampler &sampler) const;

	void addEnergyDeriv(image::SmallImage &img, image::SmallImage &dSigmaT,
						image::SmallImage &dAlbedo, image::SmallImage &dGVal,
						const VectorType<Float> &p, const VectorType<Float> &d,
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

	inline const VectorType<Float>& getRefrDir() const {
		return m_refrDir;
	}

	inline const Block<VectorType>& getMediumBlock() const {
		return m_block;
	}

	inline const AreaSource<VectorType>& getAreaSource() const {
		return m_source;
	}

//	inline const AreaTexturedSource<VectorType>& getAreaSource() const {
//		return m_source;
//	}

	inline const Camera<VectorType>& getCamera() const {
		return m_camera;
	}

	inline const bsdf::SmoothDielectric<VectorType>& getBSDF() const {
		return m_bsdf;
	}

	~Scene() { }

protected:
	Float m_ior;
	Float m_fresnelTrans;
	VectorType<Float> m_refrDir;
	Block<VectorType> m_block;
	AreaSource<VectorType> m_source;
//	AreaTexturedSource<VectorType> m_source;
	Camera<VectorType> m_camera;
	bsdf::SmoothDielectric<VectorType> m_bsdf;
	US<VectorType> m_us;
};

}	/* namespace scn */

#endif /* SCENE_H_ */
