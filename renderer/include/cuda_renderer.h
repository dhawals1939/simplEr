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

namespace cuda {

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

private:

    void setup(image::SmallImage& target, int numPhotons);
    void cleanup();
    void genDeviceRandomNumbers(int num, CudaSeedType seed = CudaSeedType(5489));
    unsigned int requiredRandomNumbers(int numPhotons);

    curandGenerator_t generator;

    /* Host memory*/
    float *image;

    /* Device memory*/
    float *cudaImage;
    float *cudaRandom;

};

}


#endif /* CUDA_RENDERER_H_ */