/*
 * cuda_renderer.h
 *
 *  Created on: Aug 2, 2021
 *      Author: Andre
 */


#ifndef CUDA_RENDERER_H_
#define CUDA_RENDERER_H_

#include <curand.h>

#include "image.h"
#include "sampler.h"

#include "cuda_scene.cuh"

namespace cuda {

// Store symbols here so as to avoid long argument list
// in kernel calls
struct Constants {
    Float *image;
    Float *random;
    Scene *scene;
    Medium *medium;
    Float weight;

    static void free(Constants c) {
        if (c.image) CUDA_CALL(cudaFree(c.image));
        if (c.random) CUDA_CALL(cudaFree(c.random));
        if (scene) Scene::free(scene);
        if (medium) Medium::free(medium);
    }
};

__constant__ Constants d_constants;

typedef unsigned int CudaSeedType;

// TODO: Consider exiting on error instead of just printing error
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    }} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    }} while(0)

class CudaRenderer {

public:

    CudaRenderer() {};

    ~CudaRenderer();

    void renderImage(image::SmallImage& target, int numPhotons);

    Constants h_constants;


private:

    void setup(image::SmallImage& target, int numPhotons);
    void cleanup();
    void genDeviceRandomNumbers(int num, CudaSeedType seed = CudaSeedType(5489));
    unsigned int requiredRandomNumbers(int numPhotons);

    curandGenerator_t generator;

    /* Host memory*/
    float *image;
};

}


#endif /* CUDA_RENDERER_H_ */
