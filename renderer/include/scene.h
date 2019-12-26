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
#include <chrono>

#include "constants.h"
#include "image.h"
#include "phase.h"
#include "pmf.h"
#include "vmf.h"
#include "warp.h"
#include "matrix.h"
#include "tvector.h"
#include "sampler.h"
#include "spline.h"
#include "tvector.h"
#include "medium.h"
#include "bsdf.h"
#include "util.h"

#include "ceres/ceres.h"

using ceres::CostFunction;
using ceres::SizedCostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

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

	AreaTexturedSource(const VectorType<Float> &origin, const VectorType<Float> &dir, const Float &halfThetaLimit, const std::string& filename,
			const tvec::Vec2f &plane, Float Li, const EmitterType &emittertype = EmitterType::directional)
			: m_origin(origin),
			  m_dir(dir),
			  m_halfThetaLimit(halfThetaLimit),
			  m_emittertype(emittertype),
			  m_plane(plane),
			  m_Li(Li) { /* m_dir(std::cos(angle), std::sin(angle), FPCONST(0.0)) */
		m_texture.readFile(filename);
		int _length = m_texture.getXRes()*m_texture.getYRes();
		m_pixelsize.x = m_plane.x/m_texture.getXRes();
		m_pixelsize.y = m_plane.y/m_texture.getYRes();

		m_ct = std::cos(m_halfThetaLimit);

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
	Float m_halfThetaLimit;
	Float m_ct;
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
    int    mode;         // Order of the bessel function or mode of the ultrasound

    VectorType<Float>    axis_uz;          // Ultrasound axis
    VectorType<Float>    axis_ux;          // Ultrasound x-axis. Need to compute angle as mode > 0 is a function of phi

    VectorType<Float>    p_u;             // A point on the ultra sound axis

    Float tol;
    Float rrWeight;
    Float invrrWeight;

    Float er_stepsize;

#ifdef SPLINE_RIF
    spline::Spline<2> m_spline;
#endif

    US(const Float& f_u, const Float& speed_u,
                 const Float& n_o, const Float& n_max, const int& mode,
                 const VectorType<Float> &axis_uz, const VectorType<Float> &axis_ux, const VectorType<Float> &p_u, const Float &er_stepsize,
				 const Float &tol, const Float &rrWeight
#ifdef SPLINE_RIF
				 , const Float xmin[], const Float xmax[],  const int N[]
#endif
				 )
#ifdef SPLINE_RIF
					 :m_spline(xmin, xmax, N)
#endif
    {
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

		this->tol 			 = tol;
		this->rrWeight       = rrWeight;
		this->invrrWeight    = 1/rrWeight;

#ifdef SPLINE_RIF
		Float *data = new Float[N[0]*N[1]];
		Float xres[2];
		xres[0] = (xmax[0] - xmin[0])/(N[0] - 1);
		xres[1] = (xmax[1] - xmin[1])/(N[1] - 1);

		VectorType<Float> p;


		for(int i=0; i < N[0]; i++)
			for(int j=0; j < N[1]; j++){
				p[0] = 0;
				p[1] = xmin[1] + xres[1] * j;
				p[2] = xmin[0] + xres[0] * i;
				data[i + j*N[0]] = bessel_RIF(p, 1) - n_o; // Only fit the varying RIF. We will add the constant later. This is to include scaling factor easily.
			}

		m_spline.build(data);
		auto end = std::chrono::steady_clock::now();
#endif
    }

    inline Float RIF(const VectorType<Float> &p, const Float &scaling) const{
#ifndef SPLINE_RIF
    	return bessel_RIF(p, scaling);
#else
    	return spline_RIF(p, scaling);
#endif
    }

    inline const VectorType<Float> dRIF(const VectorType<Float> &q, const Float &scaling) const{
#ifndef SPLINE_RIF
    	return bessel_dRIF(q, scaling);
#else
    	return spline_dRIF(q, scaling);
#endif
    }

    inline const Matrix3x3 HessianRIF(const VectorType<Float> &p, const Float &scaling) const{
#ifndef SPLINE_RIF
    	return bessel_HessianRIF(p, scaling);
#else
    	return spline_HessianRIF(p, scaling);
#endif
    }

#ifdef SPLINE_RIF
    inline double spline_RIF(const VectorType<Float> &p, const Float &scaling) const{
    	Float temp[2];
    	temp[0] = p.y;
    	temp[1] = p.z;
    	return (m_spline.value<0, 0>(temp)*scaling + n_o);
    }

    inline const VectorType<Float> spline_dRIF(const VectorType<Float> &q, const Float &scaling) const{
    	Float temp[2];
    	temp[0] = q.z;
    	temp[1] = q.y;

//    	return scaling*m_spline.gradient2d(temp);
    	return scaling*VectorType<Float>(0.0, m_spline.value<0, 1>(temp), m_spline.value<1, 0>(temp));
    }

    inline const Matrix3x3 spline_HessianRIF(const VectorType<Float> &p, const Float &scaling) const{
    	Float temp[2];
    	temp[0] = p.z;
    	temp[1] = p.y;

//    	return scaling*m_spline.hessian2d(temp);
    	Float hxy = m_spline.value<1, 1>(temp);
        return scaling*Matrix3x3(0, 0,   0,
        				 0, m_spline.value<0, 2>(temp), hxy,
    					 0, hxy, 						m_spline.value<2, 0>(temp));

    }
#endif

    inline double bessel_RIF(const VectorType<Float> &p, const Float &scaling) const;

    inline const VectorType<Float> bessel_dRIF(const VectorType<Float> &q, const Float &scaling) const;

    inline const Matrix3x3 bessel_HessianRIF(const VectorType<Float> &p, const Float &scaling) const;

    inline const Float getStepSize() const{return er_stepsize;}

    inline const Float getTol() const{return tol;}

    inline const Float getrrWeight() const{return rrWeight;}

    inline const Float getInvrrWeight() const{return invrrWeight;}

};

template <template <typename> class VectorType>
class Scene;

template <template <typename> class VectorType>
class NEECostFunction: public SizedCostFunction<3, 3>
{
	public:
	NEECostFunction(const Scene<VectorType>* refrScene, const VectorType<Float> &p1, const VectorType<Float> &p2, const Matrix3x3 &dpdv0, const Matrix3x3 &dvdv0, const Float &scaling) {
		m_refrScene = refrScene;
		m_p1 = p1;
		m_p2 = p2;
		m_dpdv0 = dpdv0;
		m_dvdv0 = dvdv0;
		m_scaling = scaling;
	}
	NEECostFunction(const Scene<VectorType>* refrScene) {
		m_refrScene = refrScene;
	}
	void updateParameters(const VectorType<Float> &p1, const VectorType<Float> &p2, const Matrix3x3 &dpdv0, const Matrix3x3 &dvdv0, const Float &scaling) {
		m_p1 = p1;
		m_p2 = p2;
		m_dpdv0 = dpdv0;
		m_dvdv0 = dvdv0;
		m_scaling = scaling;
	}
	virtual ~NEECostFunction(){}

	virtual bool Evaluate(double const* const* parameters,
						  double* residuals,
						  double** jacobians) const{

		VectorType<Float> v_i;
		VectorType<Float> p1 = m_p1;
		VectorType<Float> p2 = m_p2;
		Matrix3x3 dpdv0 = m_dpdv0;
		Matrix3x3 dvdv0 = m_dvdv0;
		Float scaling = m_scaling;

		VectorType<Float> error;
		Matrix3x3 derror;

		v_i[0] = parameters[0][0];
		v_i[1] = parameters[0][1];
		v_i[2] = parameters[0][2];
		if(m_refrScene == NULL){
			std::cerr << "Scene pointer is NULL; terminating the runs";
			std::exit(EXIT_FAILURE);
		}
		m_refrScene->computefdfNEE(v_i, p1, p2, dpdv0, dvdv0, scaling, error, derror);

		residuals[0] = error.x;
		residuals[1] = error.y;
		residuals[2] = error.z;
		if (jacobians != NULL && jacobians[0] != NULL){
			jacobians[0][0] = derror.m[0][0];
			jacobians[0][1] = derror.m[0][1];
			jacobians[0][2] = derror.m[0][2];
			jacobians[0][3] = derror.m[1][0];
			jacobians[0][4] = derror.m[1][1];
			jacobians[0][5] = derror.m[1][2];
			jacobians[0][6] = derror.m[2][0];
			jacobians[0][7] = derror.m[2][1];
			jacobians[0][8] = derror.m[2][2];
		}
		return true;
	}

	private:
	const Scene<VectorType> *m_refrScene;

	VectorType<Float> m_p1;
	VectorType<Float> m_p2;

	Matrix3x3 m_dpdv0;
	Matrix3x3 m_dvdv0;
	Float m_scaling;
};


template <template <typename> class VectorType>
class Scene {
public:
	Scene(Float ior,
			const VectorType<Float> &blockL,
			const VectorType<Float> &blockR,
			const VectorType<Float> &lightOrigin,
			const VectorType<Float> &lightDir,
			const Float &halfThetaLimit,
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
			const Float &er_stepsize,
			const Float &tol, const Float &rrWeight
#ifdef SPLINE_RIF
			, const Float xmin[], const Float xmax[],  const int N[]
#endif
            ) :
				m_ior(ior),
				m_fresnelTrans(FPCONST(1.0)),
				m_refrDir(),
				m_block(blockL, blockR),
#ifndef PROJECTOR
				m_source(lightOrigin, lightDir, lightPlane, Li),
#else                
				m_source(lightOrigin, lightDir, halfThetaLimit, lightTextureFile, lightPlane, Li),
#endif
				m_camera(viewOrigin, viewDir, viewHorizontal, viewPlane, pathlengthRange),
				m_bsdf(FPCONST(1.0), ior),
				m_us(f_u, speed_u, n_o, n_max, mode, axis_uz, axis_ux, p_u, er_stepsize, tol, rrWeight
#ifdef SPLINE_RIF
						, xmin, xmax, N
#endif
						){

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

		//Adithya: Move these to the command line or atleast the important ones
		m_options.check_gradients = false;
		m_options.max_num_iterations = 100;
		m_options.minimizer_type = ceres::LINE_SEARCH;
		m_options.line_search_direction_type = ceres::BFGS;
        m_options.logging_type = ceres::SILENT;
		m_options.function_tolerance = 1e-18;
		m_options.gradient_tolerance = 0.0;
		m_options.parameter_tolerance = 0.0;
		m_options.minimizer_progress_to_stdout = false;
	}

	/*
	 * ER trackers
	 */

    inline const Float getUSFrequency() const{
    	return m_us.f_u;
    }

	inline VectorType<Float> dP(const VectorType<Float> d) const{ // assuming omega tracking
		return d;
	}

	inline VectorType<Float> dV(const VectorType<Float> &p, const VectorType<Float> &d, const Float &scaling) const{
		return m_us.dRIF(p, scaling);
	}

	//Adi: Clean this code after bugfix
	inline Matrix3x3 d2Path(const VectorType<Float> &p, const VectorType<Float> &v, const Matrix3x3 &dpdv0, const Matrix3x3 &dvdv0, const Float &scaling) const{
		Float n = m_us.RIF(p, scaling);
		Matrix3x3 t(v, m_us.dRIF(p, scaling));
		t = t*dpdv0;
		t = -1/(n*n) * t;
		t += 1/n * dvdv0;
		return t;
	}

	inline Matrix3x3 d2V(const VectorType<Float> &p, const VectorType<Float> &v, const Matrix3x3 &dpdv0, const Float &scaling) const{
		return m_us.HessianRIF(p, scaling)*dpdv0;
	}

	inline VectorType<Float> dOmega(const VectorType<Float> p, const VectorType<Float> d, const Float &scaling) const{
		VectorType<Float> dn = m_us.dRIF(p, scaling);
		Float              n = m_us.RIF(p, scaling);

		return (dn - dot(d, dn)*d)/n;
	}

	inline void er_step(VectorType<Float> &p, VectorType<Float> &d, const Float &stepSize, const Float &scaling) const;
	inline void er_derivativestep(VectorType<Float> &p, VectorType<Float> &v, Matrix3x3 &dpdv0, Matrix3x3 &dvdv0, const Float &stepSize, const Float &scaling) const;

	inline void trace(VectorType<Float> &p, VectorType<Float> &d, const Float &distance, const Float &scaling) const; // Non optical
	inline void traceTillBlock(VectorType<Float> &p, VectorType<Float> &d, const Float &dist, Float &disx, Float &disy, Float &totalOpticalDistance, const Float &scaling) const;
	inline void trace_optical_distance(VectorType<Float> &p, VectorType<Float> &d, const Float &distance, const Float &scaling) const; // optical

	/* makeSurfaceDirectConnection: Makes direct connection by optimizing the direction to the sensor.
	 * distTravelled is unaffected if we are using simplified timing or, is updated with the total new optical path length.
	 * The dirTosensor is the optimal normalized direction found by the optimization algorithm.
	 * distToSensor is the geometric distance of the optimized connections
	 * Weight is the total weight that accounts for russian roulette passed runs, sampling of the initial velocity?
	 * Returns if surface connection has succeeded or failed
	 */
	bool makeSurfaceDirectConnection(const VectorType<Float> &p1, const VectorType<Float> &p2, const Float &scaling, smp::Sampler &sampler,
															Float &distTravelled, VectorType<Float> &dirToSensor, Float &distToSensor, Float &weight,
															scn::NEECostFunction<VectorType> &costFunction, Problem &problem, Float *initialization) const;

	void computePathLengthstillZ(const VectorType<Float> &v, const VectorType<Float> &p1, const VectorType<Float> &p2, Float &opticalPathLength, Float &t_l, const Float &scaling) const;

	void computefdfNEE(const VectorType<Float> &v_i, const VectorType<Float> &p1, const VectorType<Float> &p2, Matrix3x3 &dpdv0, Matrix3x3 &dvdv0, const Float &scaling, VectorType<Float> &error, Matrix3x3 &derror) const;
	/*
	 * TODO: Inline these methods in implementations.
	 */
	bool movePhotonTillSensor(VectorType<Float> &p, VectorType<Float> &d, Float &distToSensor, Float &totalOpticalDistance,
					smp::Sampler &sampler, const Float &scaling) const;
	bool movePhoton(VectorType<Float> &p, VectorType<Float> &d, Float dist, Float &totalOpticalDistance,
					smp::Sampler &sampler, const Float &scaling) const;
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

	//distTravelled is optical distance if USE_SIMPLIFIED_TIMING is not set and geometric distance if not
	void addEnergyInParticle(image::SmallImage &img, const VectorType<Float> &p,
						const VectorType<Float> &d, Float distTravelled, Float val,
						const med::Medium &medium, smp::Sampler &sampler, const Float &scaling) const;

	void addEnergy(image::SmallImage &img, const VectorType<Float> &p,
						const VectorType<Float> &d, Float distTravelled, Float val,
						const med::Medium &medium, smp::Sampler &sampler, const Float& scaling,
						scn::NEECostFunction<VectorType> &costFunction, Problem &problem, Float *initialization) const;

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
    inline Float getMediumIor(const VectorType<Float> &p, const Float &scaling) const {
        return m_us.RIF(p, scaling);
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

#ifndef PROJECTOR
	inline const AreaSource<VectorType>& getAreaSource() const {
		return m_source;
	}
#else
	inline const AreaTexturedSource<VectorType>& getAreaSource() const {
		return m_source;
	}
#endif

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
#ifndef PROJECTOR
	AreaSource<VectorType> m_source;
#else
	AreaTexturedSource<VectorType> m_source;
#endif
	Camera<VectorType> m_camera;
	bsdf::SmoothDielectric<VectorType> m_bsdf;
public:
	US<VectorType> m_us;

	Solver::Options m_options;
};

}	/* namespace scn */

#endif /* SCENE_H_ */
