/*
 * cuda_renderer.h
 *
 *  Created on: Aug 2, 2021
 *      Author: Andre
 */


#ifndef CUDA_RENDERER_H_
#define CUDA_RENDERER_H_

#include "image.h"
#include "curand.h"
#include "sampler.h"

typedef unsigned int CudaSeedType;

class CudaRenderer {

public:

    CudaRenderer(image::SmallImage& target, int numPhotons)
        : target(target), \
          numPhotons(numPhotons){
        setup();
    }

    virtual ~CudaRenderer();

    void renderImage();

    /* For testing */
    void compareRNGTo(smp::Sampler sampler, int numSamples);

private:

    void setup();
    void genDeviceRandomNumbers(int num, CudaSeedType seed = CudaSeedType(5489));
    unsigned int requiredRandomNumbers();

    image::SmallImage& target;
    int numPhotons;


    /* Host memory*/
    float *image;

    /* Device memory*/
    float *cudaImage;
    float *cudaRandom;

};

#endif /* CUDA_RENDERER_H_ */
