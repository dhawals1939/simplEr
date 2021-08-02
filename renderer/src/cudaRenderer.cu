#include "cudaRenderer.h"
#include "image.h"

/* TODO:
 * - The contribution of a photon equates to a sum on the pixel(s) it affects. Figure
 * out how to synchronize the contributions across all photons.
 * - Random number generation has to match that of the boost library. In the future,
 * support for the sse rng as well. */

// Store symbols here so as to avoid long argument list
// in kernel calls
struct Constants {
    float* image;
};

__constant__ Constants constants;

__global__ void renderPhotons() {
    //int i = threadIdx.x;

}

void CudaRenderer::renderImage(image::SmallImage& img, int numPhotons) {

    setup(img);

    renderPhotons<<<1,numPhotons>>>();
    cudaDeviceSynchronize();

}

void CudaRenderer::setup(image::SmallImage& img) {

    // use smallImage.x, y and z
    cudaMalloc(&cudaImage, sizeof(float) * img.getXRes() * img.getYRes() * img.getZRes());

    Constants params;
    params.image = cudaImage;

    cudaMemcpyToSymbol(constants, &params, sizeof(Constants));

}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (cudaImage) {
        cudaFree(cudaImage);
    }

    // No need to free constant memory
}
