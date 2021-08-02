#include "image.h"

class CudaRenderer {

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    void renderImage(image::SmallImage& img, int numPhotons);

private:

    void setup(image::SmallImage& img);

    /* Host memory*/
    float *image;

    /* Device memory*/
    float *cudaImage;

};
