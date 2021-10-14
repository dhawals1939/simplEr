#ifndef CUDA_SCENE_H_
#define CUDA_SCENE_H_

#include "cuda_vector.cuh"
#include "cuda_image.cuh"
#include "cuda_utils.cuh"

#include "constants.h"
#include "scene.h"


// Rendering a photon requires 6 calls to sampler() or
// equivalently, 6 random floats/doubles (FIXME: Not really true)
#define RANDOM_NUMBERS_PER_PHOTON 6

namespace cuda {

class DiscreteDistribution {
public:
    /* Return a discreteDistribution on the GPU. */
    __host__ static DiscreteDistribution* from(const image::Image2<Float> &texture, size_t length) {
        DiscreteDistribution result = DiscreteDistribution(texture, length);
        DiscreteDistribution *d_result;

        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(DiscreteDistribution)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(DiscreteDistribution), cudaMemcpyHostToDevice));
        return d_result;
    }

    __host__ DiscreteDistribution(const image::Image2<Float> &texture, size_t length) {
        m_cdf_length = length;
        m_cdf_capacity = 0;
        reserve(length);
        clear();
		for(int i=0; i<length; i++){
			append(texture.getPixel(i));
		}
		normalize();

        // Copy m_cdf to GPU
        Float *temp;
        CUDA_CALL(cudaMalloc((void **)&temp, (length+1) * sizeof(Float)));
        CUDA_CALL(cudaMemcpy(temp, m_cdf, (length+1) * sizeof(Float), cudaMemcpyHostToDevice));
        free(m_cdf);
        m_cdf = temp;
    }

	__device__ explicit inline DiscreteDistribution(size_t nEntries = 0) {
        m_cdf_length = 0;
        m_cdf_capacity = 0;
		reserve(nEntries);
        clear();
	}

	__host__ __device__ inline void clear() {
		m_cdf_length = 0;
		append(0.0f);
		m_normalized = false;
	}

    // TODO: This might be inneficient
	__host__ __device__ inline void reserve(size_t nEntries) {
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

    // TODO: Is this efficient?
	__host__ __device__ inline void append(Float pdfValue) {
        if (m_cdf_length == m_cdf_capacity) {
            reserve(m_cdf_capacity ? m_cdf_capacity * 2 : 1);
        }
        m_cdf[m_cdf_length] = m_cdf[m_cdf_length-1] + pdfValue;
        m_cdf_length++;
        ASSERT(m_cdf_length <= m_cdf_capacity);
	}
//
//	inline size_t size() const {
//		return m_cdf.size()-1;
//	}
//
	__device__ inline Float operator[](size_t entry) const {
		return m_cdf[entry+1] - m_cdf[entry];
	}

	__host__ __device__ inline bool isNormalized() const {
		return m_normalized;
	}
//
//	inline Float getSum() const {
//		return m_sum;
//	}
//
//	inline Float getNormalization() const {
//		return m_normalization;
//	}
//
	__host__ __device__ inline Float normalize() {
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

	__device__ inline size_t sample(Float sampleValue) const {
        // First elements is dummy 0.0f. Account for it.
        size_t index = fminf(m_cdf_length-2,
                           fmaxf(cdf_lower_bound(sampleValue)-1, (size_t) 0));

		/* Handle a rare corner-case where a entry has probability 0
		   but is sampled nonetheless */
		while (operator[](index) == 0 && index < m_cdf_length-1)
			++index;

		return index;
	}
//
//	inline size_t sample(Float sampleValue, Float &pdf) const {
//		size_t index = sample(sampleValue);
//		pdf = operator[](index);
//		return index;
//	}
//
//	inline size_t sampleReuse(Float &sampleValue) const {
//		size_t index = sample(sampleValue);
//		sampleValue = (sampleValue - m_cdf[index])
//			/ (m_cdf[index + 1] - m_cdf[index]);
//		return index;
//	}
//
//	inline size_t sampleReuse(Float &sampleValue, Float &pdf) const {
//		size_t index = sample(sampleValue, pdf);
//		sampleValue = (sampleValue - m_cdf[index])
//			/ (m_cdf[index + 1] - m_cdf[index]);
//		return index;
//	}

private:
	Float *m_cdf;
    size_t m_cdf_length;
    size_t m_cdf_capacity;

    __host__ __device__ size_t cdf_lower_bound(Float value) const {
        for (size_t i=0; i < m_cdf_length; i++) {
            if (m_cdf[i] >= value) return i;
        }
        return m_cdf_length;
    }

	Float m_sum, m_normalization;
	bool m_normalized;
};

class Sampler {
public:

    /* Random should be a host pointer */
    __host__ static Sampler *from(const Float *d_random, size_t size) {
        Sampler result = Sampler(d_random, size);
        Sampler *d_result;

        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(Sampler)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(Sampler), cudaMemcpyHostToDevice));
        return d_result;
    }

    /* Sample a random number for this thread and update uses argument for bookkeeping. */
    __device__ Float sample(short &uses) const;

private:

    __host__ Sampler(const Float *d_random, size_t size) : m_random(d_random), m_size(size) { }
    size_t m_size;
    const Float *m_random;
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

    __device__ inline bool propagateTillLens(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) const {
        Float dist = -(pos[0] - (*m_origin)[0])/dir[0];            //FIXME: Assumes that the direction of propagation is in -x direction.
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

    __device__ inline const TVector3<Float>& getOrigin() const {
        return *m_origin;
    }

protected:
    __host__ Lens(const scn::Lens<tvec::TVector3> &lens) {
        m_origin = TVector3<Float>::from(lens.getOrigin());
        m_squareApertureRadius = lens.getSquareApertureRadius();
        m_focalLength = lens.getFocalLength();
        m_active = lens.isActive();
    }

    TVector3<Float> *m_origin;
    Float m_squareApertureRadius; //radius of the aperture
    Float m_focalLength;
    bool m_active; // Is the lens present or absent
};

// Currently only supporting AreaTexturedSource (which assumes PROJECTOR flag is on
// for non-CUDA code).
class AreaTexturedSource {

public:

    typedef scn::AreaTexturedSource<tvec::TVector3>::EmitterType EmitterType;

    /* Create a copy AreaTexturedSource on the GPU. */
    __host__ static AreaTexturedSource *from(const scn::AreaTexturedSource<tvec::TVector3> &source) {
        AreaTexturedSource result = AreaTexturedSource(source);
        AreaTexturedSource *d_result;
        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(AreaTexturedSource)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(AreaTexturedSource), cudaMemcpyHostToDevice));
        return d_result;
    }

	__device__ bool sampleRay(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance, Sampler *sampler, short &samplerUses) const;

	__device__ inline const TVector3<Float>& getOrigin() const {
		return *m_origin;
	}

	__device__ inline const TVector3<Float>& getDir() const {
		return *m_dir;
	}

	__device__ inline const TVector2<Float>& getPlane() const {
		return *m_plane;
	}

	__device__ inline Float getLi() const {
		return m_Li;
	}

	__device__ inline bool propagateTillMedium(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) const{
		//propagate till lens
		return m_lens->propagateTillLens(pos, dir, totalDistance);
	}

	__host__ virtual ~AreaTexturedSource() { }

protected:
	__host__ AreaTexturedSource(const scn::AreaTexturedSource<tvec::TVector3> &source)
        : m_emittertype(source.getEmitterType()) {
        m_origin         = TVector3<Float>::from(source.getOrigin());
        m_dir            = TVector3<Float>::from(source.getDir());
        m_texture        = Image2<Float>::from(source.getTexture());
        m_lens           = Lens::from(source.getLens());
        m_plane          = TVector2<Float>::from(source.getPlane());
        m_Li             = source.getLi();
        m_halfThetaLimit = source.getHalfThetaLimit();

		size_t _length = source.getTexture().getXRes()*source.getTexture().getYRes();
        tvec::TVector2<Float> pixelsize(m_plane->x/m_texture->getXRes(),
                                        m_plane->y/m_texture->getYRes());
        m_pixelsize = TVector2<Float>::from(pixelsize);

		m_ct = cosf(m_halfThetaLimit);

        m_textureSampler = DiscreteDistribution::from(source.getTexture(), _length);

	}

	TVector3<Float> *m_origin;
	TVector3<Float> *m_dir;
	Float m_halfThetaLimit;
	Float m_ct;
    Image2<Float> *m_texture;
	DiscreteDistribution *m_textureSampler;
	TVector2<Float> *m_pixelsize;
	TVector2<Float> *m_plane;
	Float m_Li;
	Lens *m_lens;
	const EmitterType m_emittertype;
};

class Scene {

public:

    __host__ static Scene *from(const scn::Scene<tvec::TVector3> &scene, const Float *d_random, size_t random_size) {
        Scene result = Scene(scene, d_random, random_size);
        Scene *d_result;
        CUDA_CALL(cudaMalloc((void **)&d_result, sizeof(Scene)));
        CUDA_CALL(cudaMemcpy(d_result, &result, sizeof(Scene), cudaMemcpyHostToDevice));
        return d_result;
    }

    __device__ inline Float getUSPhi_min() const{
    	return m_us_phi_min;
    }

    __device__ inline Float getUSPhi_range() const{
    	return m_us_phi_range;
    }

    __device__ inline Float getUSMaxScaling() const{
    	return m_us_max_scaling;
    }

    __device__ bool genRay(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance, short &samplerUses) {
        return m_source->sampleRay(pos, dir, totalDistance, m_sampler, samplerUses);
    }

    __device__ Float sample(short &uses) const{
        return m_sampler->sample(uses);
    }

private:

    __host__ Scene(const scn::Scene<tvec::TVector3> &scene, const Float *d_random, size_t random_size) {
        m_source = AreaTexturedSource::from(scene.getAreaSource());
        m_sampler = Sampler::from(d_random, random_size);

        // FIXME: Implement US<TVector3> in CUDA code. This also assumes scene is constant.
        m_us_phi_min = scene.getUSPhi_min();
        m_us_phi_range = scene.getUSPhi_range();
        m_us_max_scaling = scene.getUSMaxScaling();
    }


	AreaTexturedSource *m_source;
    Float m_us_phi_min;
    Float m_us_phi_range;
    Float m_us_max_scaling;
    Sampler *m_sampler;
};



struct Medium {

    __host__ static Medium *from(const med::Medium &medium) {
        Medium result = Medium(medium.getSigmaT(), medium.getAlbedo());
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

	//inline const pfunc::HenyeyGreenstein *getPhaseFunction() const {
	//	return m_phase;
	//}

	__host__ virtual ~Medium() { }

protected:
    __host__ Medium(const Float sigmaT, const Float albedo)//, pfunc::HenyeyGreenstein *phase)
		: m_sigmaT(sigmaT),
		  m_albedo(albedo),
		  m_sigmaS(albedo * sigmaT),
		  m_sigmaA((1 - albedo) * sigmaT),
		  m_mfp(FPCONST(1.0) / sigmaT)
		  //m_phase(phase)
    {
		ASSERT(m_sigmaA >= 0);
		ASSERT(m_albedo <= 1);
		if (m_sigmaT <= M_EPSILON) {
			m_sigmaT = FPCONST(0.0);
			m_mfp = FPCONST(1.0);
			m_albedo = FPCONST(0.0);
		}
	}

	Float m_sigmaT;
	Float m_albedo;
	Float m_sigmaS;
	Float m_sigmaA;
	Float m_mfp;
	//pfunc::HenyeyGreenstein *m_phase;
};

}


#endif // CUDA_SCENE_H_
