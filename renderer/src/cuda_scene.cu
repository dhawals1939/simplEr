#include "cuda_scene.cuh"

__device__ Float Sampler::sample(short &uses) const{
    ASSERT(uses < RANDOM_NUMBERS_PER_PHOTON);
    ASSERT(threadIdx.x * RANDOM_NUMBERS_PER_PHOTON + uses + 1 < m_size);
    return m_random[threadIdx.x * RANDOM_NUMBERS_PER_PHOTON + uses++];
}

// Sample random ray
__device__ bool AreaTexturedSource::sampleRay(TVector3<Float> &pos, TVector3<Float> &dir,
                                      Float &totalDistance, Sampler *sampler, short &samplerUses) const{
    pos = m_origin;

    // sample pixel position first
	int pixel = m_textureSampler.sample(sampler.sample(samplerUses));
	int p[2];
	m_texture.ind2sub(pixel, p[0], p[1]);

	// Now find a random location on the pixel
	for (int iter = 1; iter < m_origin.dim; ++iter) {
		pos[iter] += - m_plane[iter - 1] / FPCONST(2.0) +
            p[iter - 1] * m_pixelsize[iter-1] + sampler.sample(samplerUses) * m_pixelsize[iter - 1];
	}

	dir = m_dir;

	//FIXME: Hack: Works only for m_dir = [-1 0 0]
	Float z   = sampler->sample(samplerUses)*(1-m_ct) + m_ct;
	Float zt  = sqrtf(FPCONST(1.0)-z*z);
    // FIXME: FPCONST(M_PI) might be generating complaints here
	Float phi = sampler->sample(samplerUses)*2*FPCONST(M_PI);
    // FIXME: operator[] overload here might be causing issues
	dir[0] = -z;
	dir[1] = zt*cosf(phi);
	dir[2] = zt*sinf(phi);

	return propagateTillMedium(pos, dir, totalDistance);
}
