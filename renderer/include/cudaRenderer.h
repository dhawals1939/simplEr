/*
 * cudaRenderer.h
 *
 *  Created on: Aug 2, 2021
 *      Author: Andre
 */


#ifndef CUDA_RENDERER_H_
#define CUDA_RENDERER_H_

#include "image.h"
#include "curand.h"


typedef unsigned int CudaSeedType;

class CudaRenderer {

public:

    CudaRenderer(image::SmallImage& img, int numPhotons)
        : img(img), \
          numPhotons(numPhotons){}

    virtual ~CudaRenderer();

    void renderImage();

private:

    void setup();
    float *genDeviceRandomNumbers(CudaSeedType seed);

    image::SmallImage& img;
    int numPhotons;


    /* Host memory*/
    float *image;

    /* Device memory*/
    float *cudaImage;
    float *cudaRandom;

};

#endif /* CUDA_RENDERER_H_ */
