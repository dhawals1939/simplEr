
#include "tvector.h"
#include <vector>
#include <stdio.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(tvector_pybind, m) {
    m.doc() = "tvector pybind module";

    py::class_<tvec::TVector2<Float> >(m, "Vec2f")
            		.def(py::init<
            				Float, Float
            		>());


	py::class_<tvec::TVector3<Float> >(m, "Vec3f")
			.def(py::init<
					Float, Float, Float
			>())
			.def("index", &tvec::TVector3<Float>::index);


	py::class_<tvec::TVector3<int> >(m, "Vec3i")
				.def(py::init<
						int, int, int
				>())
				.def_readwrite("x", &tvec::TVector3<int>::x)
				.def_readwrite("y", &tvec::TVector3<int>::y)
				.def_readwrite("z", &tvec::TVector3<int>::z);

}

