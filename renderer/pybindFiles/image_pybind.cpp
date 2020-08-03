
#include "image.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(image_pybind, m) {
    m.doc() = "image pybind module";

    py::class_<image::Image2<Float> > smallImage(m, "SmallImage",  py::buffer_protocol());

    smallImage.def(py::init<int, int>());
	smallImage.def("getXRes", &image::Image2<Float>::getXRes);
	smallImage.def("getYRes", &image::Image2<Float>::getYRes);
	smallImage.def("readFile", &image::Image2<Float>::readFile);
    smallImage.def("getPixel", (Float (image::Image2<Float>::*)(const int, const int) const) &image::Image2<Float>::getPixel);
    py::enum_<image::Image2<Float>::EFileFormat>(smallImage, "EFileFormat")
        .value("EOpenEXR",image::Image2<Float>::EOpenEXR)
        .value("EPFM",image::Image2<Float>::EPFM)
        .value("EFileFormatLength",image::Image2<Float>::EFileFormatLength)
        .value("EFileFormatInvalid",image::Image2<Float>::EFileFormatInvalid)
        ;
}
