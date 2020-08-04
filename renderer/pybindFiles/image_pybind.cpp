
#include "image.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(image_pybind, m) {
    m.doc() = "image pybind module";

    py::class_<image::Image3<Float> > smallImage(m, "SmallImage",  py::buffer_protocol());

    smallImage.def(py::init<int, int, int>());
	smallImage.def("getXRes", &image::Image3<Float>::getXRes);
	smallImage.def("getYRes", &image::Image3<Float>::getYRes);
	smallImage.def("getZRes", &image::Image3<Float>::getZRes);
	smallImage.def("readPFM3D", &image::Image3<Float>::readPFM3D);
    smallImage.def("getPixel", (Float (image::Image3<Float>::*)(const int, const int, const int) const) &image::Image3<Float>::getPixel);
    py::enum_<image::Image2<Float>::EFileFormat>(smallImage, "EFileFormat")
        .value("EOpenEXR",image::Image2<Float>::EOpenEXR)
        .value("EPFM",image::Image2<Float>::EPFM)
        .value("EFileFormatLength",image::Image2<Float>::EFileFormatLength)
        .value("EFileFormatInvalid",image::Image2<Float>::EFileFormatInvalid)
        ;
}
