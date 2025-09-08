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

namespace scn
{

    template <template <typename> class vector_type>
    struct US
    {
        Float wavelength_u; // (m)
        Float R;            // Radius of the ultrasound wavefront (m)
        Float inv_R2;       // 1/R^2

#if USE_RIF_SOURCES
        Float f_u;
        Float speed_u;
        Float n_o;
        Float n_scaling;
        int n_coeff;
        Float radius;
        //    Float[] center;
        vector_type<Float> center1;
        vector_type<Float> center2;
        bool active1;
        bool active2;
        Float phase1;
        Float phase2;
        Float theta_min;
        Float theta_max;
        int theta_sources;
        Float trans_z_min;
        Float trans_z_max;
        int trans_z_sources;
        int nsources;
        vector_type<Float> *centers1;
        vector_type<Float> *centers2;
#else
        Float f_u;          // Ultrasound frequency (1/s or Hz)
        Float speed_u;      // Ultrasound speed (m/s)
        Float n_o;          // Baseline refractive index
        Float n_max;        // Max refractive index variation
        Float n_clip;       // Clipped refractive index variation
        Float n_maxScaling; // =n_clip/n_max
        Float phi_min;      // Min Phase
        Float phi_max;      // Max Phase
        int mode;           // Order of the bessel function or mode of the ultrasound
#endif

        Float k_r;

        vector_type<Float> axis_uz; // Ultrasound axis
        vector_type<Float> axis_ux; // Ultrasound x-axis. Need to compute angle as mode > 0 is a function of phi

        vector_type<Float> p_u; // A point on the ultra sound axis

        Float tol;
        Float rrWeight;
        Float invrrWeight;

        Float er_stepsize;
        int m_precision;
        Float m_EgapEndLocX;
        Float m_SgapBeginLocX;

        bool m_useInitializationHack;

#if USE_RIF_INTERPOLATED
        //    spline::Spline<2> m_spline;
        spline::Spline<3> m_spline;
#endif

        US(
#if USE_RIF_SOURCES
            const Float &f_u, const Float &speed_u, const Float &n_o, const Float &n_scaling, const Float &n_coeff, const Float &radius, const vector_type<Float> &center1, const vector_type<Float> &center2, const bool &active1, const bool &active2, const Float &phase1, const Float &phase2, const Float &theta_min, const Float &theta_max, const int &theta_sources, const Float &trans_z_min, const Float &trans_z_max, const int &trans_z_sources,
#else
            const Float &f_u, const Float &speed_u, const Float &n_o, const Float &n_max, const Float &n_clip, const Float &phi_min, const Float &phi_max, const int &mode,
#endif
            const vector_type<Float> &axis_uz, const vector_type<Float> &axis_ux, const vector_type<Float> &p_u, const Float &er_stepsize,
            const Float &tol, const Float &rrWeight, const int &precision, const Float &EgapEndLocX, const Float &SgapBeginLocX, const bool &useInitializationHack
#if USE_RIF_INTERPOLATED
            //               , const Float xmin[], const Float xmax[],  const int N[]
            ,
            const std::string &rifgridFile
#endif
            )
#if USE_RIF_INTERPOLATED
            //                   :m_spline(xmin, xmax, N)
            : m_spline(rifgridFile)
#endif
        {
#if USE_RIF_SOURCES
            this->f_u = f_u;
            this->speed_u = speed_u;
            this->n_o = n_o;
            this->n_scaling = n_scaling;
            this->n_coeff = n_coeff;
            this->radius = radius;
            this->center1 = center1;
            this->center2 = center2;
            this->active1 = active1;
            this->active2 = active2;
            this->phase1 = phase1;
            this->phase2 = phase2;
            this->theta_min = theta_min;
            this->theta_max = theta_max;
            this->theta_sources = theta_sources;
            this->trans_z_min = trans_z_min;
            this->trans_z_max = trans_z_max;
            this->trans_z_sources = trans_z_sources;

            Float theta_diff = (theta_max - theta_min) / (theta_sources - 1);
            Float trans_z_diff = (trans_z_max - trans_z_min) / (trans_z_sources - 1);

            this->nsources = theta_sources * trans_z_sources;
            if (this->active1)
                this->centers1 = new vector_type<Float>[this->nsources];
            if (this->active2)
                this->centers2 = new vector_type<Float>[this->nsources];

            for (int i = 0; i < this->theta_sources; i++)
            {
                Float theta = this->theta_min + theta_diff * i;
                Float xval = this->radius * (1 - cos(theta));
                Float yval = this->radius * (sin(theta));

                for (int j = 0; j < this->trans_z_sources; j++)
                {
                    // int index = i*trans_z_sources + j;
                    int index = i + j * this->theta_sources; // to match matlab indices and debug
                    // for horizontal (0, 0, 0.0508)
                    if (this->active1)
                    {
                        this->centers1[index].y = yval + this->center1.y;
                        this->centers1[index].z = xval + this->center1.z;
                        this->centers1[index].x = this->trans_z_min + trans_z_diff * j + this->center1.x;
                    }
                    // for vertical (0, -0.0508, 0)
                    if (this->active2)
                    {
                        this->centers2[index].y = xval + this->center2.y;
                        this->centers2[index].z = yval + this->center2.z;
                        this->centers2[index].x = this->trans_z_min + trans_z_diff * j + this->center2.x;
                    }
                }
            }
#else
            this->f_u = f_u;
            this->speed_u = speed_u;
            this->n_o = n_o;
            this->n_max = n_max;
            this->n_clip = n_clip;
            this->n_maxScaling = n_clip / n_max;
            this->phi_min = phi_min;
            this->phi_max = phi_max;
            this->mode = mode;
#endif
            this->wavelength_u = ((double)this->speed_u) / this->f_u;
            this->k_r = (2 * M_PI) / this->wavelength_u;
            this->R = this->wavelength_u / 2.;
            this->inv_R2 = 1 / this->R * this->R;
            this->axis_uz = axis_uz;
            this->axis_ux = axis_ux;
            this->p_u = p_u;

            this->er_stepsize = er_stepsize;

            this->tol = tol;
            this->rrWeight = rrWeight;
            this->invrrWeight = 1 / rrWeight;
            this->m_precision = precision;
            this->m_EgapEndLocX = EgapEndLocX;
            this->m_SgapBeginLocX = SgapBeginLocX;

            this->m_useInitializationHack = useInitializationHack;
        }

        inline Float RIF(const vector_type<Float> &p, const Float &scaling) const
        {
            if (p.x > this->m_EgapEndLocX || p.x < this->m_SgapBeginLocX)
                return this->n_o;
#if USE_RIF_SOURCES
            return this->fus_RIF(p, scaling);
#elif USE_RIF_PARABOLIC
            return this->parabolic_RIF(p, scaling);
#elif USE_RIF_INTERPOLATED
            return this->spline_RIF(p, scaling);
#else
            return this->bessel_RIF(p, scaling);
#endif
        }

        inline const vector_type<Float> dRIF(const vector_type<Float> &p, const Float &scaling) const
        {
            if (p.x > this->m_EgapEndLocX || p.x < this->m_SgapBeginLocX)
                return vector_type<Float>(0.0);
#if USE_RIF_SOURCES
            return this->fus_dRIF(p, scaling);
#elif USE_RIF_PARABOLIC
            return this->parabolic_dRIF(p, scaling);
#elif USE_RIF_INTERPOLATED
            return this->spline_dRIF(p, scaling);
#else
            return this->bessel_dRIF(p, scaling);     
#endif
        }

        inline const Matrix3x3 HessianRIF(const vector_type<Float> &p, const Float &scaling) const
        {
            if (p.x > this->m_EgapEndLocX || p.x < this->m_SgapBeginLocX)
                return Matrix3x3(0.0);
#if USE_RIF_SOURCES
            return this->fus_HessianRIF(p, scaling);
#elif PARABOLIC_RIF
            return this->parabolic_HessianRIF(p, scaling);
#elif USE_RIF_INTERPOLATED
            return this->spline_HessianRIF(p, scaling);
#else
            return this->bessel_HessianRIF(p, scaling);
#endif
        }


#if USE_RIF_SOURCES

        inline double fus_RIF(const vector_type<Float> &p, const Float &scaling) const;

        inline const vector_type<Float> fus_dRIF(const vector_type<Float> &q, const Float &scaling) const;

        inline const Matrix3x3 fus_HessianRIF(const vector_type<Float> &p, const Float &scaling) const;
#elif USE_RIF_INTERPOLATED
        inline double spline_RIF(const vector_type<Float> &p, const Float &scaling) const
        {
            Float temp[3];
            temp[0] = p.x;
            temp[1] = p.y;
            temp[2] = p.z;
            return this->m_spline.value(temp);
            //      Float temp[2];
            //      temp[0] = p.y;
            //      temp[1] = p.z;
            //      return (m_spline.value<0, 0>(temp)*scaling + n_o);
        }

        inline const vector_type<Float> spline_dRIF(const vector_type<Float> &p, const Float &scaling) const
        {
            Float temp[3];
            temp[0] = p.x;
            temp[1] = p.y;
            temp[2] = p.z;
            return this->m_spline.gradient(temp);
            //              vector_type<Float>(m_spline.value<1, 0, 0>(temp), m_spline.value<0, 1, 0>(temp), m_spline.value<0, 0, 1>(temp));
            //      Float temp[2];
            //      temp[0] = q.z;
            //      temp[1] = q.y;
            //
            ////        return scaling*m_spline.gradient2d(temp);
            //      return scaling*vector_type<Float>(0.0, m_spline.value<0, 1>(temp), m_spline.value<1, 0>(temp));
        }

        inline const Matrix3x3 spline_HessianRIF(const vector_type<Float> &p, const Float &scaling) const
        {
            Float temp[3];
            temp[0] = p.x;
            temp[1] = p.y;
            temp[2] = p.z;
            return this->m_spline.hessian(temp);
            //      Float temp[2];
            //      temp[0] = p.z;
            //      temp[1] = p.y;
            //
            ////        return scaling*m_spline.hessian2d(temp);
            //      Float hxy = m_spline.value<1, 1>(temp);
            //        return scaling*Matrix3x3(0, 0,   0,
            //                       0, m_spline.value<0, 2>(temp), hxy,
            //                       0, hxy,                        m_spline.value<2, 0>(temp));
        }
#elif PARABOLIC_RIF

        inline double parabolic_RIF(const vector_type<Float> &p, const Float &scaling) const;

        inline const vector_type<Float> parabolic_dRIF(const vector_type<Float> &q, const Float &scaling) const;

        inline const Matrix3x3 parabolic_HessianRIF(const vector_type<Float> &p, const Float &scaling) const;
#else

        inline double bessel_RIF(const vector_type<Float> &p, const Float &scaling) const;

        inline const vector_type<Float> bessel_dRIF(const vector_type<Float> &q, const Float &scaling) const;

        inline const Matrix3x3 bessel_HessianRIF(const vector_type<Float> &p, const Float &scaling) const;

#endif

        inline const Float getStepSize() const { return this->er_stepsize; }

        inline const Float getTol2() const { return this->tol * this->tol; }

        inline const Float getrrWeight() const { return this->rrWeight; }

        inline const Float getInvrrWeight() const { return this->invrrWeight; }

        inline const int getPrecision() const { return this->m_precision; }
    };

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

     inline Float getMediumIor() const
     {
         return m_ior;
     }
     inline Float getMediumIor(const vector_type<Float> &p, const Float &scaling) const
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
     inline const area_source<vector_type> &getAreaSource() const
     {
         return m_source;
     }
#else
        inline const area_textured_source<vector_type> &getAreaSource() const
        {
            return m_source;
        }
#endif

     inline const Camera<vector_type> &getCamera() const
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

