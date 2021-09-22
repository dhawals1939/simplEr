#import "cuda_scene.cuh"
#include "math.h"

// Sample random ray
__device__ bool AreaTexturedSource::sampleRay(TVector3<Float> &pos, TVector3<Float> &dir,
                                      Float &totalDistance, Float (*sampler)(void)) {
	pos = m_origin;

	// sample pixel position first
	int pixel = m_textureSampler.sample(sampler());
	int p[2];
	m_texture.ind2sub(pixel, p[0], p[1]);

	// Now find a random location on the pixel
	for (int iter = 1; iter < m_origin.dim; ++iter) {
		pos[iter] += - m_plane[iter - 1] / FPCONST(2.0) +
            p[iter - 1] * m_pixelsize[iter-1] + sampler() * m_pixelsize[iter - 1];
	}

	dir = m_dir;

	//FIXME: Hack: Works only for m_dir = [-1 0 0]
	Float z   = sampler()*(1-m_ct) + m_ct;
	Float zt  = sqrt(FPCONST(1.0)-z*z);
	Float phi = sampler()*2*FPCONST(M_PI);
	dir[0] = -z;
	dir[1] = zt*cos(phi);
	dir[2] = zt*sin(phi);

	return propagateTillMedium(pos, dir, totalDistance);
}
