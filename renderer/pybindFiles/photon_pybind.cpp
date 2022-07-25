
#include "photon.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(photon_pybind, m) {
    m.doc() = "photon pybind module";


	py::class_<photon::Renderer<tvec::TVector3> >(m, "Renderer")
			.def(py::init<
				const int, const Float, const bool, const bool, const int64_t
			>())
			.def("renderImage", &photon::Renderer<tvec::TVector3>::renderImage);

}
