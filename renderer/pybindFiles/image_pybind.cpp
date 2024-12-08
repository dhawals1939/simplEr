
#include "image.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(image_pybind, m) {
    m.doc() = "image pybind module";

    py::class_<image::Image3<double> > (m, "SmallImage",  py::buffer_protocol())
		.def(py::init<
				int, int, int
		>())
		.def("getXRes", &image::Image3<double>::getXRes)
		.def("getYRes", &image::Image3<double>::getYRes)
		.def("getZRes", &image::Image3<double>::getZRes)
		.def("getPixel", &image::Image3<double>::getPixel)
//		.def("getData", &image::Image3<double>::getData)
//        .def_buffer([](image::Image3<double> &m) -> py::buffer_info {
//	        return py::buffer_info(
//	            m.getData(),                               /* Pointer to buffer */
//	            sizeof(double),                          /* Size of one scalar */
//	            py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
//	            3,                                      /* Number of dimensions */
//	            { m.getXRes(), m.getYRes(), m.getZRes() },                 /* Buffer dimensions */
//	            { sizeof(double), sizeof(double) * m.getYRes(),             /* Strides (in bytes) for each index */
//	              sizeof(double)*m.getYRes()*m.getXRes() }
//	        );
//        })
		;
}
