/*
 * scene.h
 *
 *  Created on: Nov 26, 2015
 *      Author: igkiou
 */

#pragma once

#include <stdio.h>
#include <vector>
#include <chrono>

#include <constants.h>
#include <image.h>
#include <phase.h>
#include <pmf.h>
#include <vmf.h>
#include <warp.h>
#include <matrix.h>
#include <tvector.h>
#include <sampler.h>
#include <spline.h>
#include <tvector.h>
#include <medium.h>
#include <bsdf.h>
#include <util.h>
#include <block.h>
#include <camera.h>
#include <lens.h>
#include <area_textured_source.h>
#include <area_source.h>
#include <omp.h>
#include <rif.h>

namespace scn
{

  
    template <template <typename> class vector_type>
    class Scene;

    template <template <typename> class vector_type>
    class Scene
    {
    public:
     Scene(Float ior,
             const vector_type<Float> &blockL,
             const vector_type<Float> &blockR,
             const vector_type<Float> &lightOrigin,
             const vector_type<Float> &lightDir,
             const Float &halfThetaLimit,
             const std::string &lightTextureFile,
             const tvec::Vec2f &lightPlane,
             const Float Li,
             const vector_type<Float> &viewOrigin,
             const vector_type<Float> &viewDir,
             const vector_type<Float> &viewHorizontal,
             const tvec::Vec2f &viewPlane,
             const tvec::Vec2f &pathlength_range, 
             const bool &use_bounce_decomposition,
             // for finalAngle importance sampling
             const std::string &distribution,
             const Float &gOrKappa,
             // for emitter lens
             const vector_type<Float> &emitter_lens_origin,
             const Float &emitter_lens_aperture,
             const Float &emitter_lens_focalLength,
             const bool &emitter_lens_active,
             // for sensor lens
             const vector_type<Float> &sensor_lens_origin,
             const Float &sensor_lens_aperture,
             const Float &sensor_lens_focalLength,
             const bool &sensor_lens_active,
     // Ultrasound parameters: a lot of them are currently not used
#if      USE_RIF_SOURCES
             const Float& f_u,
             const Float& speed_u,
             const Float& n_o,
             const Float& n_scaling,
             const int& n_coeff,
             const Float& radius,
             const vector_type<Float> &center1,
             const vector_type<Float> &center2,
             const bool &active1,
             const bool &active2,
             const Float &phase1,
             const Float &phase2,
             const Float& theta_min,
             const Float& theta_max,
             const int& theta_sources,
             const Float& trans_z_min,
             const Float& trans_z_max,
             const int& trans_z_sources,
#else
              const Float &f_u,
              const Float &speed_u,
              const Float &n_o,
              const Float &n_max,
              const Float &n_clip,
              const Float &phi_min,
              const Float &phi_max,
              const int &mode,
#endif
             const vector_type<Float> &axis_uz,
             const vector_type<Float> &axis_ux,
             const vector_type<Float> &p_u,
             const Float &er_stepsize,
             const Float &tol, const Float &rrWeight, const int &precision, const Float &EgapEndLocX, const Float &SgapBeginLocX, const bool &useInitializationHack
#if USE_RIF_INTERPOLATED
 //          , const Float xmin[], const Float xmax[],  const int N[]
             , const std::string &rifgridFile
#endif
             ) :
                 m_ior(ior),
                 m_fresnelTrans(FPCONST(1.0)),
                 m_refrDir(),
                 m_block(blockL, blockR),
#ifndef PROJECTOR
                 m_source(lightOrigin, lightDir, lightPlane, Li),
#else
                  m_source(lightOrigin, lightDir, halfThetaLimit, lightTextureFile, lightPlane, Li, emitter_lens_origin, emitter_lens_aperture, emitter_lens_focalLength, emitter_lens_active),
#endif
                 m_camera(viewOrigin, viewDir, viewHorizontal, viewPlane, pathlength_range, use_bounce_decomposition, sensor_lens_origin, sensor_lens_aperture, sensor_lens_focalLength, sensor_lens_active),
                 m_bsdf(FPCONST(1.0), ior),
#if USE_RIF_SOURCES
                 m_us(f_u, speed_u, n_o, n_scaling, n_coeff, radius, center1, center2, active1, active2, phase1, phase2, theta_min, theta_max, theta_sources, trans_z_min, trans_z_max, trans_z_sources, axis_uz, axis_ux, p_u, er_stepsize, tol, rrWeight, precision, EgapEndLocX, SgapBeginLocX, useInitializationHack) // Need to fix this
#else
                  m_us(f_u, speed_u, n_o, n_max, n_clip, phi_min, phi_max, mode, axis_uz, axis_ux, p_u, er_stepsize, tol, rrWeight, precision, EgapEndLocX, SgapBeginLocX, useInitializationHack
#endif
#if USE_RIF_INTERPOLATED
                        //  , xmin, xmax, N
                         , rifgridFile)
#else
#ifndef USE_RIF_SOURCES
                  )
#endif
#endif
     {

         Assert(((std::abs(m_source.get_origin().x - m_block.get_block_l().x) < M_EPSILON) && (m_source.get_dir().x > FPCONST(0.0))) ||
                ((std::abs(m_source.get_origin().x - m_block.get_block_r().x) < M_EPSILON) && (m_source.get_dir().x < FPCONST(0.0))));

         if (m_ior > FPCONST(1.0))
         {
             Float sumSqr = FPCONST(0.0);
             for (int iter = 1; iter < m_refrDir.dim; ++iter)
             {
                 m_refrDir[iter] = m_source.get_dir()[iter] / m_ior;
                 sumSqr += m_refrDir[iter] * m_refrDir[iter];
             }
             m_refrDir.x = std::sqrt(FPCONST(1.0) - sumSqr);
             if (m_source.get_dir().x < FPCONST(0.0))
             {
                 m_refrDir.x *= -FPCONST(1.0);
             }
#ifndef USE_NO_FRESNEL
             m_fresnelTrans = m_ior * m_ior * (FPCONST(1.0) - util::fresnelDielectric(m_source.get_dir().x, m_refrDir.x, m_ior));
#endif
         }
         else
         {
             m_refrDir = m_source.get_dir();
         }
#if USE_PRINTING
         std::cout << "fresnel " << m_fresnelTrans << std::endl;
#endif
     }

     /*
      * ER trackers
      */

     inline const Float getUSFrequency() const
     {
         return m_us.f_u;
     }

     inline void set_f_u(Float &value)
     {
         m_us.f_u = value;
     }

#if USE_RIF_SOURCES
     inline const Float getUSPhi_min() const
     {
         return 0.;
     }

     inline const Float getUSPhi_range() const
     {
         return 0.;
     }

     inline const Float getUSMaxScaling() const
     {
         return 0.;
     }

     inline void set_n_scaling(Float &value)
     {
         m_us.n_scaling = value;
     }

     inline void set_phase1(Float &value)
     {
         m_us.phase1 = value;
     }

     inline void set_phase2(Float &value)
     {
         m_us.phase2 = value;
     }

#else
        inline void set_n_max(Float &value)
        {
            m_us.n_max = value;
        }

        inline Float get_RIF(vector_type<Float> &p, Float &scaling)
        {
            return m_us.RIF(p, scaling);
        }

        inline vector_type<Float> get_dRIF(vector_type<Float> &p, Float &scaling)
        {
            return m_us.dRIF(p, scaling);
        }

        inline const Float getUSPhi_min() const
        {
            return m_us.phi_min;
        }

        inline const Float getUSPhi_range() const
        {
            return (m_us.phi_max - m_us.phi_min);
        }

        inline const Float getUSMaxScaling() const
        {
            return m_us.n_maxScaling;
        }

#endif

     inline vector_type<Float> dP(const vector_type<Float> d) const
     { // assuming omega tracking
         return d;
     }

     inline vector_type<Float> dV(const vector_type<Float> &p, const vector_type<Float> &d, const Float &scaling) const
     {
         return m_us.dRIF(p, scaling);
     }

     // Adi: Clean this code after bugfix
     inline Matrix3x3 d2Path(const vector_type<Float> &p, const vector_type<Float> &v, const Matrix3x3 &dpdv0, const Matrix3x3 &dvdv0, const Float &scaling) const
     {
         Float n = m_us.RIF(p, scaling);
         Matrix3x3 t(v, m_us.dRIF(p, scaling));
         t = t * dpdv0;
         t = -1 / (n * n) * t;
         t += 1 / n * dvdv0;
         return t;
     }

     inline Matrix3x3 d2V(const vector_type<Float> &p, const vector_type<Float> &v, const Matrix3x3 &dpdv0, const Float &scaling) const
     {
         return m_us.HessianRIF(p, scaling) * dpdv0;
     }

     inline vector_type<Float> dOmega(const vector_type<Float> p, const vector_type<Float> d, const Float &scaling) const
     {
         vector_type<Float> dn = m_us.dRIF(p, scaling);
         Float n = m_us.RIF(p, scaling);

         return (dn - dot(d, dn) * d) / n;
     }

     inline void er_step(vector_type<Float> &p, vector_type<Float> &d, const Float &stepSize, const Float &scaling) const;
     inline void er_derivativestep(vector_type<Float> &p, vector_type<Float> &v, Matrix3x3 &dpdv0, Matrix3x3 &dvdv0, const Float &stepSize, const Float &scaling) const;

     inline void trace(vector_type<Float> &p, vector_type<Float> &d, const Float &distance, const Float &scaling) const; // Non optical
     inline void traceTillBlock(vector_type<Float> &p, vector_type<Float> &d, const Float &dist, Float &disx, Float &disy, Float &totalOpticalDistance, const Float &scaling) const;
     inline void trace_optical_distance(vector_type<Float> &p, vector_type<Float> &d, const Float &distance, const Float &scaling) const; // optical

     void computePathLengthstillZ(const vector_type<Float> &v, const vector_type<Float> &p1, const vector_type<Float> &p2, Float &opticalPathLength, Float &t_l, const Float &scaling) const;

     void computefdfNEE(const vector_type<Float> &v_i, const vector_type<Float> &p1, const vector_type<Float> &p2, Matrix3x3 &dpdv0, Matrix3x3 &dvdv0, const Float &scaling, vector_type<Float> &error, Matrix3x3 &derror) const;
     /*
      * TODO: Inline these methods in implementations.
      */
     bool movePhotonTillSensor(vector_type<Float> &p, vector_type<Float> &d, Float &distToSensor, Float &totalOpticalDistance,
                               smp::Sampler &sampler, const Float &scaling) const;
     bool movePhoton(vector_type<Float> &p, vector_type<Float> &d, Float dist, Float &totalOpticalDistance,
                     smp::Sampler &sampler, const Float &scaling) const;
     bool genRay(vector_type<Float> &pos, vector_type<Float> &dir, smp::Sampler &sampler, Float &total_distance) const;
     bool genRay(vector_type<Float> &pos, vector_type<Float> &dir, smp::Sampler &sampler,
                 vector_type<Float> &possrc, vector_type<Float> &dirsrc, Float &total_distance) const;
     void addEnergyToImage(image::SmallImage &img, const vector_type<Float> &p,
                           Float pathlength, int &depth, Float val) const;

     inline void addPixel(image::SmallImage &img, int x, int y, int z, Float val) const
     {
         if (x >= 0 && x < img.get_x_res() && y >= 0 && y < img.get_y_res() &&
             z >= 0 && z < img.getZRes())
         {
             img.addEnergy(x, y, z, static_cast<Float>(val));
         }
     }

     // distTravelled is optical distance if USE_SIMPLIFIED_TIMING is not set and geometric distance if not
     void addEnergyInParticle(image::SmallImage &img, const vector_type<Float> &p,
                              const vector_type<Float> &d, Float distTravelled, int &depth, Float val,
                              const med::Medium &medium, smp::Sampler &sampler, const Float &scaling) const;

     void addEnergyDeriv(image::SmallImage &img, image::SmallImage &dSigmaT,
                         image::SmallImage &dAlbedo, image::SmallImage &dGVal,
                         const vector_type<Float> &p, const vector_type<Float> &d,
                         Float distTravelled, Float val, Float sumScoreSigmaT,
                         Float sumScoreAlbedo, Float sumScoreGVal,
                         const med::Medium &medium, smp::Sampler &sampler) const;

     /*
      * TODO: Direct lighting is currently not supported.
      */
     //  void addEnergyDirect(image::SmallImage &img, const tvec::Vec3f &p,
     //                      const tvec::Vec3f &d, Float val,
     //                      const med::Medium &medium, smp::Sampler &sampler) const;

     inline Float get_medium_ior() const
     {
         return m_ior;
     }
     inline Float get_medium_ior(const vector_type<Float> &p, const Float &scaling) const
     {
         return m_us.RIF(p, scaling);
     }

     inline Float getFresnelTrans() const
     {
         return m_fresnelTrans;
     }

     inline const vector_type<Float> &getRefrDir() const
     {
         return m_refrDir;
     }

     inline const block<vector_type> &getMediumBlock() const
     {
         return m_block;
     }

#ifndef PROJECTOR
     inline const area_source<vector_type> &get_area_source() const
     {
         return m_source;
     }
#else
        inline const area_textured_source<vector_type> &get_area_source() const
        {
            return m_source;
        }
#endif

     inline const Camera<vector_type> &get_camera() const
     {
         return m_camera;
     }

     inline const bsdf::SmoothDielectric<vector_type> &getBSDF() const
     {
         return m_bsdf;
     }

     ~Scene() {}

 protected:
     Float m_ior;
     Float m_fresnelTrans;
     vector_type<Float> m_refrDir;
     block<vector_type> m_block;
#ifndef PROJECTOR
     area_source<vector_type> m_source;
#else
        area_textured_source<vector_type> m_source;
#endif
     Camera<vector_type> m_camera;
     bsdf::SmoothDielectric<vector_type> m_bsdf;

 public:
     US<vector_type> m_us;

    };

} /* namespace scn */

