
#include "phase.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(phase_pybind, m) {
    m.doc() = "phase pybind module";

    py::class_<pfunc::henyey_greenstein>(m, "henyey_greenstein")
			.def(py::init<Float>());
}
