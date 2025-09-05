
#include "medium.h"
#include "phase.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;


PYBIND11_MODULE(medium_pybind, m) {
    m.doc() = "medium pybind module";

    py::class_<med::Medium>(m, "Medium")
			.def(py::init<
					Float, Float, pfunc::henyey_greenstein*
			>());

}
