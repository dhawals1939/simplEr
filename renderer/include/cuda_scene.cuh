#ifndef CUDA_SCENE_H_
#define CUDA_SCENE_H_

#include "cuda_vector.cuh"
#include "cuda_image.cuh"
#include "cuda_utils.cuh"

#include "constants.h"
#include "scene.h"

namespace cuda {

class DiscreteDistribution {
public:
    /* Return a discreteDistribution on the GPU. */
    __host__ static DiscreteDistribution* from(const std::vector<Float> &cdf) {
        DiscreteDistribution result = DiscreteDistribution(cdf.data(), cdf.size());
        DiscreteDistribution *d_result;

        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(DiscreteDistribution)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(DiscreteDistribution), cudaMemcpyHostToDevice));
        return d_result;
    }

    __device__ inline void clear() {
        m_cdf_length = 0;
        append(0.0f);
        m_normalized = false;
    }

    __device__ inline void reserve(size_t nEntries) {
        ASSERT(nEntries >= m_cdf_capacity);
        m_cdf_capacity = nEntries+1;
        Float *temp = (Float *)malloc((nEntries+1) * sizeof(Float));
        if (m_cdf) {
            memcpy(temp, m_cdf, m_cdf_length * sizeof(Float));
            free(m_cdf);
        }
        m_cdf = temp;
        ASSERT(m_cdf);
    }

    __device__ inline void append(Float pdfValue) {
        if (m_cdf_length == m_cdf_capacity) {
            reserve(m_cdf_capacity ? m_cdf_capacity * 2 : 1);
        }
        if (m_cdf_length == 0)
            m_cdf[m_cdf_length] = pdfValue;
        else
            m_cdf[m_cdf_length] = m_cdf[m_cdf_length-1] + pdfValue;

        m_cdf_length++;
        ASSERT(m_cdf_length <= m_cdf_capacity);
    }

    __device__ inline Float operator[](size_t entry) const {
        return m_cdf[entry+1] - m_cdf[entry];
    }

    __device__ inline bool isNormalized() const {
        return m_normalized;
    }

    __device__ inline Float normalize() {
        ASSERT(m_cdf_length > 1);
        m_sum = m_cdf[m_cdf_length-1];
        if (m_sum > 0) {
            m_normalization = 1.0f / m_sum;
            for (size_t i=1; i<m_cdf_length; ++i)
                m_cdf[i] *= m_normalization;
            m_cdf[m_cdf_length-1] = 1.0f;
            m_normalized = true;
        } else {
            m_normalization = 0.0f;
        }
        return m_sum;
    }

    __device__ inline size_t sample(Float sampleValue) {
        // First elements is dummy 0.0f. Account for it.
        size_t index = fminf(m_cdf_length-2,
                           fmaxf(cdf_lower_bound(sampleValue)-1, (size_t) 0));

        /* Handle a rare corner-case where a entry has probability 0
           but is sampled nonetheless */
        while (operator[](index) == 0 && index < m_cdf_length-1)
            ++index;

        return index;
    }

private:
    Float *m_cdf;
    size_t m_cdf_length;
    size_t m_cdf_capacity;

    /* cdf is a pointer to the host */
    __host__ DiscreteDistribution(const Float *cdf, size_t length) {
        CUDA_CALL(cudaMalloc((void **)&m_cdf, length * sizeof(Float)));
        CUDA_CALL(cudaMemcpy(m_cdf, cdf, length * sizeof(Float), cudaMemcpyHostToDevice));
        m_cdf_length = length;
        m_cdf_capacity = length;
        m_normalized = false;
    }

    __device__ inline size_t cdf_lower_bound(Float value) const {
        size_t lo = 0;
        size_t hi = m_cdf_length;
        size_t mid;
        while (lo < hi) {
            mid = lo + (hi - lo)/2;
            if (value < m_cdf[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    Float m_sum, m_normalization;
    bool m_normalized;
};

class SmoothDielectric {
public:
    __host__ static SmoothDielectric *from(const bsdf::SmoothDielectric<tvec::TVector3> &in) {
        SmoothDielectric result = SmoothDielectric(in);
        SmoothDielectric *d_result;

        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(SmoothDielectric)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(SmoothDielectric), cudaMemcpyHostToDevice));
        return d_result;
    }

    __device__ void sample(const TVector3<Float> &in, const TVector3<Float> &n,
                TVector3<Float> &out) const;

    __device__ inline Float getIor1() const {
        return m_ior1;
    }

    __device__ inline Float getIor2() const {
        return m_ior2;
    }
private:
    Float m_ior1;
    Float m_ior2;

    __host__ SmoothDielectric(Float ior1, Float ior2) :
        m_ior1(ior1),
        m_ior2(ior2) { }

    __host__ SmoothDielectric(const bsdf::SmoothDielectric<tvec::TVector3> &in) {
        m_ior1 = in.getIor1(); m_ior2 = in.getIor2();
    }
};


class Lens {

public:
    __host__ static Lens *from(const scn::Lens<tvec::TVector3> &lens) {
        Lens result = Lens(lens);
        Lens *d_result;

        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(Lens)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(Lens), cudaMemcpyHostToDevice));
        return d_result;
    }

    // TODO: Enable and implement free for other cuda classes
    // Frees a lens on the device (and all contained fields)
    //__host__ static void free(Lens *d_lens) {
    //    Lens *h_lens;
    //    h_result = (Lens *)malloc(sizeof(Lens));
    //    CUDA_CALL(cudaMemcpy(h_result, d_lens, sizeof(Lens)));
    //    CUDA_CALL(cudaFree(d_lens));
    //    TVector3<Float>::free(h_lens->m_origin);
    //    free(h_result);
    //}

    __device__ inline bool deflect(const TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) const {
        /* Deflection computation:
         * Point going through the center of lens and parallel to dir is [pos.x, 0, 0]. Ray from this point goes straight
         * This ray meets focal plane at (pos.x - d[0] * f/d[0], -d[1] * f/d[0], -d[2] * f/d[0]) (assuming -x direction of propagation of light)
         * Original ray deflects to pass through this point
         * The negative distance (HACK) travelled by this ray at the lens is -f/d[0] - norm(focalpoint_Pos - original_Pos)
         */
        Float squareDistFromLensOrigin = 0.0f;
        for(int i = 1; i < pos.dim; i++)
            squareDistFromLensOrigin += pos[i]*pos[i];
        if(squareDistFromLensOrigin > m_squareApertureRadius)
            return false;

        ASSERT(pos.x == m_origin->x);
        Float invd = -1/dir[0];
        dir[0] = -m_focalLength;
        dir[1] = dir[1]*invd*m_focalLength - pos[1];
        dir[2] = dir[2]*invd*m_focalLength - pos[2];
        totalDistance += m_focalLength*invd - dir.length();
        dir.normalize();
        return true; // should return additional path length added by the lens.
    }

    __device__ inline bool propagate_till_lens(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) const {
        Float dist = -(pos[0] - (*m_origin)[0])/dir[0];
        pos += dist*dir;
        totalDistance += dist;
        if(m_active)
            return deflect(pos, dir, totalDistance);
        else
            return true;
    }

    __device__ inline bool isActive() const {
        return m_active;
    }

    __device__ inline const TVector3<Float>& get_origin() const {
        return *m_origin;
    }

protected:
    __host__ Lens(const scn::Lens<tvec::TVector3> &lens) {
        m_origin = TVector3<Float>::from(lens.get_origin());
        m_squareApertureRadius = lens.getSquareApertureRadius();
        m_focalLength = lens.getFocalLength();
        m_active = lens.isActive();
    }

    TVector3<Float> *m_origin;
    Float m_squareApertureRadius; //radius of the aperture
    Float m_focalLength;
    bool m_active; // Is the lens present or absent
};

class Camera {
public:

    __host__ static Camera *from(const scn::Camera<tvec::TVector3>& camera) {
        Camera result = Camera(camera);
        Camera *d_result;

        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(Camera)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(Camera), cudaMemcpyHostToDevice));
        return d_result;
    }

    __device__ inline bool samplePosition(TVector3<Float> &pos) const;

    __device__ inline const TVector3<Float>& get_origin() const {
        return *m_origin;
    }

    __device__ inline const TVector3<Float>& get_dir() const {
        return *m_dir;
    }

    __device__ inline const TVector3<Float>& getHorizontal() const {
        return *m_horizontal;
    }

    __device__ inline const TVector3<Float>& getVertical() const {
        return *m_vertical;
    }

    __device__ inline const TVector2<Float>& get_plane() const {
        return *m_plane;
    }

    __device__ inline const TVector2<Float>& getPathlengthRange() const {
        return *m_pathlengthRange;
    }

    __device__ inline const bool& isBounceDecomposition() const {
        return m_useBounceDecomposition;
    }

    __device__ inline const bool propagateTillSensor(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) const{
        //propagate till lens
        if (m_lens->isActive() && !m_lens->deflect(pos, dir, totalDistance))
            return false;
        //propagate from lens to sensor
        Float dist = ((*m_origin)[0]-pos[0])/dir[0];            //FIXME: Assumes that the direction of propagation is in -x direction.
        pos += dist*dir;
#if PRINT_DEBUGLOG
        if (dist < -1e-4){
            std::cout << "Propagation till sensor failed; dying" << std::endl;
            exit(EXIT_FAILURE);
        }
#endif

        totalDistance += dist;
        return true;
    }

    __host__ ~Camera() { }

private:

    __host__ Camera(const scn::Camera<tvec::TVector3>& camera) {

        m_origin     = TVector3<Float>::from(camera.get_origin());
        m_dir        = TVector3<Float>::from(camera.get_dir());
        m_horizontal = TVector3<Float>::from(camera.getHorizontal());
        m_vertical   = TVector3<Float>::from(camera.getVertical());

        m_plane            = TVector2<Float>::from(camera.get_plane());
        m_pathlengthRange  = TVector2<Float>::from(camera.getPathlengthRange());

        m_useBounceDecomposition = camera.isBounceDecomposition();

        m_lens = Lens::from(camera.get_lens());
    }
    TVector3<Float> *m_origin;
    TVector3<Float> *m_dir;
    TVector3<Float> *m_horizontal;
    TVector3<Float> *m_vertical;
    TVector2<Float> *m_plane;
    TVector2<Float> *m_pathlengthRange;
    bool m_useBounceDecomposition;
    Lens *m_lens;
};

// Currently only supporting area_textured_source (which assumes PROJECTOR flag is on
// for non-CUDA code).
class area_textured_source {

public:

    typedef scn::area_textured_source<tvec::TVector3>::EmitterType EmitterType;

    /* Create a copy area_textured_source on the GPU. */
    __host__ static area_textured_source *from(const scn::area_textured_source<tvec::TVector3> &source) {
        area_textured_source result = area_textured_source(source);
        area_textured_source *d_result;
        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(area_textured_source)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(area_textured_source), cudaMemcpyHostToDevice));
        return d_result;
    }

    __device__ bool sample_ray(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) const;

    __device__ inline const TVector3<Float>& get_origin() const {
        return *m_origin;
    }

    __device__ inline const TVector3<Float>& get_dir() const {
        return *m_dir;
    }

    __device__ inline const TVector2<Float>& get_plane() const {
        return *m_plane;
    }

    __device__ inline Float get_Li() const {
        return m_Li;
    }

    __device__ inline bool propagate_till_medium(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) const{
        //propagate till lens
        return m_lens->propagate_till_lens(pos, dir, totalDistance);
    }

    __host__ virtual ~area_textured_source() { }

protected:
    __host__ area_textured_source(const scn::area_textured_source<tvec::TVector3> &source)
        : m_emittertype(source.get_emitter_type()) {
        m_origin         = TVector3<Float>::from(source.get_origin());
        m_dir            = TVector3<Float>::from(source.get_dir());
        m_texture        = Image2<Float>::from(source.get_texture());
        m_lens           = Lens::from(source.get_lens());
        m_plane          = TVector2<Float>::from(source.get_plane());
        m_Li             = source.get_Li();
        m_half_theta_limit = source.get_half_theta_limit();

        float textureXRes = source.get_texture().get_x_res();
        float textureYRes = source.get_texture().get_y_res();
        float planeX = source.get_plane().x;
        float planeY = source.get_plane().y;
        tvec::TVector2<Float> pixelsize(planeX/textureXRes,
                                        planeY/textureYRes);
        m_pixelsize = TVector2<Float>::from(pixelsize);

        m_ct = cosf(m_half_theta_limit);

        m_texture_sampler = DiscreteDistribution::from(source.texture_sampler_cdf());

    }

    TVector3<Float> *m_origin;
    TVector3<Float> *m_dir;
    Float m_half_theta_limit;
    Float m_ct;
    Image2<Float> *m_texture;
    DiscreteDistribution *m_texture_sampler;
    TVector2<Float> *m_pixelsize;
    TVector2<Float> *m_plane;
    Float m_Li;
    Lens *m_lens;
    const EmitterType m_emittertype;
};

class US {
public:
    Float    wavelength_u; // (m)
#if USE_RIF_SOURCES
    Float f_u;
    Float speed_u;
    Float n_o;
    Float n_scaling;
    int n_coeff;
    Float radius;
//    Float[] center;
    TVector3<Float> *center1;
    TVector3<Float> *center2;
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
    TVector3<Float> *centers1;
    TVector3<Float> *centers2;
#else
    Float f_u;          // Ultrasound frequency (1/s or Hz)
    Float speed_u;      // Ultrasound speed (m/s)
    Float n_o;          // Baseline refractive index
    Float n_max;        // Max refractive index variation
    Float n_clip;        // Clipped refractive index variation
    Float n_maxScaling;  // =n_clip/n_max
    Float phi_min;        // Min Phase 
    Float phi_max;        // Max Phase 
    int   mode;         // Order of the bessel function or mode of the ultrasound
#endif

    Float k_r;

    TVector3<Float> *axis_uz;          // Ultrasound axis
    TVector3<Float> *axis_ux;          // Ultrasound x-axis. Need to compute angle as mode > 0 is a function of phi

    TVector3<Float> *p_u;             // A point on the ultra sound axis

    Float tol;
    Float rrWeight;
    Float invrrWeight;

    Float er_stepsize;
    int m_precision;
    Float m_EgapEndLocX;
    Float m_SgapBeginLocX;

    bool m_useInitializationHack;


    __device__ inline Float RIF(const TVector3<Float> &p, const Float &scaling) const{
        if(p.x > m_EgapEndLocX || p.x < m_SgapBeginLocX)
            return n_o;
#if USE_RIF_SOURCES
        return fus_RIF(p, scaling);
#else
        return bessel_RIF(p, scaling);
#endif
    }

    __device__ inline const TVector3<Float> dRIF(const TVector3<Float> &p, const Float &scaling) const{
        if(p.x > m_EgapEndLocX || p.x < m_SgapBeginLocX)
            return TVector3<Float>(0.0);
#if USE_RIF_SOURCES
        return fus_dRIF(p, scaling);
#else
        return bessel_dRIF(p, scaling);
#endif
    }

    //inline const Matrix3x3 HessianRIF(const TVector3<Float> &p, const Float &scaling) const{
    //    if(p.x > m_EgapEndLocX || p.x < m_SgapBeginLocX)
    //        return Matrix3x3(0.0);
    //  return bessel_HessianRIF(p, scaling);
    //}

#if USE_RIF_SOURCES
    __device__ inline double fus_RIF(const TVector3<Float> &p, Float scaling) const{
        Float rif=0;
        for(int c=1; c<=n_coeff; c++){
            Float scalingfactor = 4*pow(-1, c+1)/(pow(c*M_PI, 2)*nsources);
            Float trif = 0.;
            if (active1){
                for(int n=0; n<nsources; n++){
                    Float r = (p - centers1[n]).length();
                    trif += cos(k_r*r*c + phase1)/r; // avoiding complex number math by taking real numbers directly
                }
            }
            if (active2){
                for(int n=0; n<nsources; n++){
                    Float r = (p - centers2[n]).length();
                    trif += cos(k_r*r*c + phase2)/r; // avoiding complex number math by taking real numbers directly
                }
            }
            rif += trif*scalingfactor;
        }
        return n_o + n_scaling * rif;
    }
#else
    __device__ inline double bessel_RIF(const TVector3<Float> &p, Float scaling) const{
        TVector3<Float> p_axis = *p_u + dot(p - *p_u, *axis_uz)*(*axis_uz); // point on the axis closest to p

        Float r    = (p-p_axis).length();
        Float dotp = dot(p-p_axis, *axis_ux);
        Float detp = dot(cross(*axis_ux, p-p_axis), *axis_uz);
        Float phi  = atan2f(detp, dotp);

        return n_o + n_max * scaling * jn(mode, k_r*r) * cosf(mode*phi);
    }
#endif

#if USE_RIF_SOURCES
    __device__ const TVector3<Float> fus_dRIF(const TVector3<Float> &p, Float scaling) const{
        TVector3<Float> dn(0.0,
                             0.0, 
                             0.0
                             );
    
        for(int c=1; c<=n_coeff; c++){
            Float scalingfactor = 4*pow(-1, c+1)/(pow(c*M_PI, 2)*nsources);
            TVector3<Float> tdn(0., 0., 0.);

            if(active1){
                for(int n=0; n<nsources; n++){
                    TVector3<Float> dirvec = p - centers1[n];
                    Float r = dirvec.length();
                    Float krrc = k_r*r*c;
                    Float commonterm = -(cos(krrc+phase1) + krrc * sin(krrc+phase1))/pow(r, 3); // avoiding complex number math by taking real numbers directly
                    tdn += commonterm * dirvec;
                }
            }
            if(active2){
                for(int n=0; n<nsources; n++){
                    TVector3<Float> dirvec = p - centers2[n];
                    Float r = dirvec.length();
                    Float krrc = k_r*r*c;
                    Float commonterm = -(cos(krrc+phase2) + krrc * sin(krrc+phase2))/pow(r, 3); // avoiding complex number math by taking real numbers directly
                    tdn += commonterm * dirvec;
                }
            }
            //std::cout << "tdn:" << tdn << std::endl;
            dn += tdn*scalingfactor;
        }
        return dn*n_scaling;
    }
#else
    __device__ const TVector3<Float> bessel_dRIF(const TVector3<Float> &q, Float scaling) const{

        TVector3<Float> p_axis = *p_u + dot(q - *p_u, *axis_uz)*(*axis_uz); // point on the axis closest to p

        TVector3<Float> p      = q - p_axis; // acts like p in case of axis aligned

        Float r    = p.length();
        Float dotp = dot(p, *axis_ux);
        Float detp = dot(cross(*axis_ux, p), *axis_uz);
        Float phi  = atan2f(detp, dotp);

        if(r < M_EPSILON)
            p.y = M_EPSILON;
        if(r < M_EPSILON)
            p.z = M_EPSILON;
        if(r < M_EPSILON)
            r = M_EPSILON;

        Float krr = k_r * r;

        Float besselj   = jn(mode, krr);

        Float dbesselj  = mode/(krr) * besselj - jn(mode+1, krr);

        Float invr  = 1.0/r;
        Float invr2 = invr * invr;

        Float cosmp  = cosf(mode * phi);
        Float sinmp  = sinf(mode * phi);

        TVector3<Float> dn(0.0,
                           n_max * scaling * (dbesselj * k_r * p.y * invr * cosmp - besselj*mode*sinmp*p.z*invr2),
                           n_max * scaling * (dbesselj * k_r * p.z * invr * cosmp + besselj*mode*sinmp*p.y*invr2));
        return dn;
    }
#endif
    //inline const TVector3<Float> bessel_dRIF(const TVector3<Float> &q, const Float &scaling) const;

    //inline const Matrix3x3 bessel_HessianRIF(const TVector3<Float> &p, const Float &scaling) const;

    __device__ inline const Float getStepSize() const{return er_stepsize;}

    __device__ inline const Float getTol2() const{return tol*tol;}

    __device__ inline const Float getrrWeight() const{return rrWeight;}

    __device__ inline const Float getInvrrWeight() const{return invrrWeight;}

    __device__ inline const int getPrecision()  const{return m_precision;}

    __host__ static US* from(scn::US<tvec::TVector3> us) {
        US result(us);
        US *d_result;

        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(US)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(US), cudaMemcpyHostToDevice));
        return d_result;
    }

protected:
    US(const scn::US<tvec::TVector3>& us) {
#if USE_RIF_SOURCES
        f_u = us.f_u;
        speed_u = us.speed_u;
        n_o = us.n_o;
        n_scaling = us.n_scaling;
        n_coeff = us.n_coeff;
        radius = us.radius;
        center1 = TVector3<Float>::from(us.center1);
        center2 = TVector3<Float>::from(us.center2);
        active1 = us.active1;
        active2 = us.active2;
        phase1 = us.phase1;
        phase2 = us.phase2;
        theta_min = us.theta_min;
        theta_max = us.theta_max;
        theta_sources = us.theta_sources;
        trans_z_min = us.trans_z_min;
        trans_z_max = us.trans_z_max;
        trans_z_sources = us.trans_z_sources;

        nsources = us.nsources;
        if(active1){
            cudaMalloc((void **)&centers1, nsources * sizeof(TVector3<Float>));
            cudaMemcpy(centers1, us.centers1, nsources * sizeof(TVector3<Float>), cudaMemcpyHostToDevice);
        }
        if(active2){
            cudaMalloc((void **)&centers2, nsources * sizeof(TVector3<Float>));
            cudaMemcpy(centers2, us.centers2, nsources * sizeof(TVector3<Float>), cudaMemcpyHostToDevice);
        }
#else
        f_u            = us.f_u;
        speed_u        = us.speed_u;
        n_o            = us.n_o;
        n_max          = us.n_max;
        n_clip         = us.n_clip;
        n_maxScaling   = us.n_maxScaling;
        phi_min        = us.phi_min;
        phi_max        = us.phi_max;
        mode           = us.mode;
#endif
        wavelength_u   = us.wavelength_u;
        k_r            = us.k_r;

        axis_uz        = TVector3<Float>::from(us.axis_uz);
        axis_ux        = TVector3<Float>::from(us.axis_ux);
        p_u            = TVector3<Float>::from(us.p_u);

        er_stepsize    = us.er_stepsize;

        tol            = us.tol;
        rrWeight       = us.rrWeight;
        invrrWeight    = us.invrrWeight;
        m_precision    = us.m_precision;
        m_EgapEndLocX  = us.m_EgapEndLocX;
        m_SgapBeginLocX= us.m_SgapBeginLocX;

        m_useInitializationHack = us.m_useInitializationHack;
    }
};

class henyey_greenstein {
public:

    __host__ static henyey_greenstein *from(const pfunc::henyey_greenstein* func) {
        henyey_greenstein result = henyey_greenstein(func->getG());
        henyey_greenstein *d_result;
        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(henyey_greenstein)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(henyey_greenstein), cudaMemcpyHostToDevice));
        return d_result;
    }

    __host__ ~henyey_greenstein() { }

    __device__ Float f(const TVector3<Float> &in, const TVector3<Float> &out) const {
        Float cosTheta = dot(in, out);
        return static_cast<Float>(FPCONST(1.0) / (4.0 * M_PI)) * (FPCONST(1.0) - m_g * m_g)
                / pow(FPCONST(1.0) + m_g * m_g - FPCONST(2.0) * m_g * cosTheta, FPCONST(1.5));
    }

    __device__ Float derivf(const TVector3<Float> &in, const TVector3<Float> &out) const {
        Float cosTheta = dot(in, out);
        Float denominator = FPCONST(1.0) + m_g * m_g - FPCONST(2.0) * m_g * cosTheta;
        return static_cast<Float>(FPCONST(1.0) / M_PI) *
            (cosTheta + cosTheta * m_g * m_g - FPCONST(2.0) * m_g)
            / denominator / denominator;
    }

    __device__ Float score(const TVector3<Float> &in, const TVector3<Float> &out) const {
        Float cosTheta = dot(in, out);
        return (cosTheta + cosTheta * m_g * m_g - FPCONST(2.0) * m_g) * FPCONST(2.0)
                / (FPCONST(1.0) - m_g * m_g)
                / (FPCONST(1.0) + m_g * m_g - FPCONST(2.0) * m_g * cosTheta);
    }


    __device__ Float sample(const TVector3<Float> &in, TVector3<Float> &out) const;

    __device__ Float sample(const TVector2<Float> &in, TVector2<Float> &out)  const;

    __device__ inline Float getG() const {
        return m_g;
    }

private:

    __device__ static inline void coordinateSystem(const TVector3<Float> &a, TVector3<Float> &b, TVector3<Float> &c) {
    if (fabsf(a.x) > fabsf(a.y)) {
        Float invLen = FPCONST(1.0) / sqrtf(a.x * a.x + a.z *a.z);
        c = TVector3<Float>(a.z * invLen, FPCONST(0.0), -a.x * invLen);
    } else {
        Float invLen = FPCONST(1.0) / sqrtf(a.y * a.y + a.z * a.z);
        c = TVector3<Float>(FPCONST(0.0), a.z * invLen, -a.y * invLen);
    }
    b = cross(c, a);
}

    __host__ henyey_greenstein(const Float g)
                    : m_g(g) {  }

    Float m_g;
};

class Medium {
public:
    __host__ static Medium *from(const med::Medium &medium) {
        Medium result = Medium(medium.getSigmaT(), medium.getAlbedo(), medium.getPhaseFunction());
        Medium *d_result;
        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(Medium)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(Medium), cudaMemcpyHostToDevice));
        return d_result;
    }

    __device__ inline Float getSigmaT() const {
        return m_sigmaT;
    }

    __device__ inline Float getSigmaS() const {
        return m_sigmaS;
    }

    __device__ inline Float getSigmaA() const {
        return m_sigmaA;
    }

    __device__ inline Float getMfp() const {
        return m_mfp;
    }

    __device__ inline Float getAlbedo() const {
        return m_albedo;
    }

    __device__ inline const henyey_greenstein *getPhaseFunction() const {
        return m_phase;
    }

    __host__ virtual ~Medium() { }

protected:
    __host__ Medium(const Float sigmaT, const Float albedo, const pfunc::henyey_greenstein *phase)
        : m_sigmaT(sigmaT),
          m_albedo(albedo),
          m_sigmaS(albedo * sigmaT),
          m_sigmaA((1 - albedo) * sigmaT),
          m_mfp(FPCONST(1.0) / sigmaT)
    {
        ASSERT(m_sigmaA >= 0);
        ASSERT(m_albedo <= 1);
        if (m_sigmaT <= M_EPSILON) {
            m_sigmaT = FPCONST(0.0);
            m_mfp = FPCONST(1.0);
            m_albedo = FPCONST(0.0);
        }
        m_phase = henyey_greenstein::from(phase);
    }

    Float m_sigmaT;
    Float m_albedo;
    Float m_sigmaS;
    Float m_sigmaA;
    Float m_mfp;
    henyey_greenstein *m_phase;
};

class Scene {

public:

    __host__ static Scene *from(const scn::Scene<tvec::TVector3> &scene) {
        Scene result = Scene(scene);
        Scene *d_result;
        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(Scene)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(Scene), cudaMemcpyHostToDevice));
        return d_result;
    }

    __device__ inline bool genRay(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) {
        return m_source->sample_ray(pos, dir, totalDistance);
    }

    __device__ inline Float getMediumIor(const TVector3<Float> &p, const Float &scaling) const {
        return m_us->RIF(p, scaling);
    }

#if USE_RIF_SOURCES
    __device__ inline Float getUSPhi_min() const{
        return 0.;
    }

    __device__ inline Float getUSPhi_range() const{
        return 0.;
    }

    __device__ inline Float getUSMaxScaling() const{
        return 0.;
    }
#else
    __device__ inline Float getUSPhi_min() const{
        return m_us->phi_min;
    }

    __device__ inline Float getUSPhi_range() const{
        return m_us->phi_max - m_us->phi_min;
    }

    __device__ inline Float getUSMaxScaling() const{
        return m_us->n_maxScaling;
    }
#endif

    __device__ void addEnergyToImage(const TVector3<Float> &p, Float pathlength, int &depth, Float val) const;

    __device__ void addEnergyInParticle(const TVector3<Float> &p, const TVector3<Float> &d, Float distTravelled,
                                        int &depth, Float val, const Float &scaling) const;

    __device__ bool movePhotonTillSensor(TVector3<Float> &p, TVector3<Float> &d, Float &distToSensor, Float &totalOpticalDistance,
                                            const Float& scaling) const;

    __device__ bool movePhoton(TVector3<Float> &p, TVector3<Float> &d, Float dist,
                               Float &totalOpticalDistance, Float scaling) const;

    __device__ void traceTillBlock(TVector3<Float> &p, TVector3<Float> &d, Float dist, Float &disx,
                                   Float &disy, Float &totalOpticalDistance, Float scaling) const;

    __device__ void er_step(TVector3<Float> &p, TVector3<Float> &d, Float stepSize, Float scaling) const;

    __device__ inline TVector3<Float> dP(const TVector3<Float> d) const{
        #ifndef OMEGA_TRACKING
        ASSERT(false);
        #endif /* OMEGA_TRACKING */
        return d;
    }

    __device__ inline TVector3<Float> dV(const TVector3<Float> &p, const TVector3<Float> &d, Float scaling) const{
        return m_us->dRIF(p, scaling);
    }

    __device__ inline TVector3<Float> dOmega(const TVector3<Float> p, const TVector3<Float> d, Float scaling) const{
        TVector3<Float> dn = m_us->dRIF(p, scaling);
        Float            n = m_us->RIF(p, scaling);

        return (dn - dot(d, dn)*d)/n;
    }

    __device__ inline const Camera &getCamera() {
        return *m_camera;
    }
private:

    __host__ Scene(const scn::Scene<tvec::TVector3> &scene) {
        m_source  = area_textured_source::from(scene.getAreaSource());
        m_us      = US::from(scene.m_us);
        m_bsdf    = SmoothDielectric::from(scene.getBSDF());
        m_camera  = Camera::from(scene.getCamera());
    }

    Camera *m_camera;
    SmoothDielectric *m_bsdf;
    US *m_us;
    area_textured_source *m_source;
};

}


#endif // CUDA_SCENE_H_
