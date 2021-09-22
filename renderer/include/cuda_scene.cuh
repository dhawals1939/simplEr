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

// Currently only supporting AreaTexturedSource (which assumes PROJECTOR flag is on
// for non-CUDA code).
class AreaTexturedSource {
public:

	enum EmitterType{directional, diffuse}; //diffuse is still not implemented

    __host__

    
	//__host__ AreaTexturedSource(const TVector3<Float> &origin, const TVector3<Float> &dir, const Float &halfThetaLimit, const std::string& filename,
	//		const TVector2<Float> &plane, const Float &Li, const TVector3<Float> &lens_origin, const Float &lens_aperture, const Float &lens_focalLength,
    //                              const bool &lens_active, const EmitterType &emittertype = EmitterType::directional)
	//		: m_origin(origin),
	//		  m_dir(dir),
	//		  m_halfThetaLimit(halfThetaLimit),
	//		  m_emittertype(emittertype),
	//		  m_plane(plane),
	//		  m_Li(Li),
	//		  m_lens(lens_origin, lens_aperture, lens_focalLength, lens_active){

    //    m_texture.readFile(filename);
	//	int _length = m_texture.getXRes()*m_texture.getYRes();
	//	m_pixelsize.x = m_plane.x/m_texture.getXRes();
	//	m_pixelsize.y = m_plane.y/m_texture.getYRes();

	//	m_ct = cos(m_halfThetaLimit);

	//	m_textureSampler.reserve(_length);
	//	for(int i=0; i<_length; i++){
	//		m_textureSampler.append(m_texture.getPixel(i));
	//	}
	//	m_textureSampler.normalize();
	//}

	__device__ bool sampleRay(TVector3<Float> &pos, TVector3<Float> &dir, Float (*sampler)(void), Float &totalDistance) const;

	__device__ inline const TVector3<Float>& getOrigin() const {
		return m_origin;
	}

	__device__ inline const TVector3<Float>& getDir() const {
		return m_dir;
	}

	__device__ inline const TVector2<Float>& getPlane() const {
		return m_plane;
	}

	__device__ inline Float getLi() const {
		return m_Li;
	}

	__device__ inline const bool propagateTillMedium(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) const{
		//propagate till lens
		return m_lens.propagateTillLens(pos, dir, totalDistance);
	}

	__host__ virtual ~AreaTexturedSource() { }

protected:
	TVector3<Float> m_origin;
	TVector3<Float> m_dir;
	Float m_halfThetaLimit;
	Float m_ct;
    Image2<Float> m_texture;
	DiscreteDistribution m_textureSampler;
	TVector2<Float> m_pixelsize;
	TVector2<Float> m_plane;
	Float m_Li;
	EmitterType m_emittertype;
	Lens<TVector3> m_lens;
}

class Scene {

public:

    __host__ Scene(Float *host_random, size_t random_size, AreaTexturedSource &source) : m_random(random), m_source(source) {
        CUDA_CALL(cudaMalloc((void **)&m_random, random_size * sizeof(Float)));
        CUDA_CALL(cudaMemcpy(m_random, host_random, random_size * sizeof(Float)));
    }

    __host__ ~Scene() {
        CUDA_CALL(cudaFree(m_random));
    }

    __device__ bool genRay(TVector3<Float> &pos, TVector3<Float> &dir, Float &totalDistance) {
        return m_source.sampleRay(pos, dir, sampler, totalDistance);
    }

private:
    __device__ Float sampler() {
        ASSERT(m_sampler_uses < RANDOM_NUMBERS_PER_PHOTON);
        return m_random[threadIdx.x * RANDOM_NUMBERS_PER_PHOTON + m_sampler_uses++];
    }

	AreaSource m_source;

    Float *m_random;
    short m_sampler_uses = 0;
};


struct DiscreteDistribution {
	__device__ explicit inline DiscreteDistribution(size_t nEntries = 0) {
        m_cdf_length = 0;
        m_cdf_capacity = 0;
		reserve(nEntries);
        //clear();
	}

	//inline void clear() {
	//	m_cdf.clear();
	//	m_cdf.push_back(0.0f);
	//	m_normalized = false;
	//}

    // TODO: This might be inneficient
	__device__ inline void reserve(size_t nEntries) {
        ASSERT(nEntries >= m_cdf_capacity);
        m_cdf_capacity = nEntries+1;
        Float *temp = malloc((nEntries+1) * sizeof(Float));
        if (m_cdf) {
            memcpy(temp, m_cdf, m_cdf_length * sizeof(Float));
            free(m_cdf);
        }
        m_cdf = temp;
        ASSERT(m_cdf);
	}

    // TODO: Is this efficient?
	__device__ inline void append(Float pdfValue) {
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
//	inline Float operator[](size_t entry) const {
//		return m_cdf[entry+1] - m_cdf[entry];
//	}
//
//	inline bool isNormalized() const {
//		return m_normalized;
//	}
//
//	inline Float getSum() const {
//		return m_sum;
//	}
//
//	inline Float getNormalization() const {
//		return m_normalization;
//	}
//
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

	__device__ inline size_t sample(Float sampleValue) const {
        // First elements is dummy 0.0f. Account for it.
        size_t index = min(m_cdf_length-2,
                           max(cdf_lower_bound(value)-1, (size_t) 0));

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

    __device__ size_t cdf_lower_bound(Float value) {
        for (size_t i=0; i < m_cdf_length; i++) {
            if (m_cdf[i] >= value) return i;
        }
        return m_cdf_length;
    }
	Float m_sum, m_normalization;
	bool m_normalized;
}

}


#endif // CUDA_SCENE_H_
