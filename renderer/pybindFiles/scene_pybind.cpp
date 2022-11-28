
#include "scene.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;


PYBIND11_MODULE(scene_pybind, m) {
    m.doc() = "scene pybind module";

    py::class_<scn::Scene<tvec::TVector3> >(m, "Scene")
			.def(py::init<
	            Float,
	            const tvec::TVector3<Float>&,
	            const tvec::TVector3<Float>&,
	            const tvec::TVector3<Float>&,
	            const tvec::TVector3<Float>&,
	            const Float&,
	            const std::string&,
	            const tvec::Vec2f&,
	            const Float,
	            const tvec::TVector3<Float>&,
	            const tvec::TVector3<Float>&,
	            const tvec::TVector3<Float>&,
	            const tvec::Vec2f&,
	            const tvec::Vec2f&, 
	            const bool&,
	            // for finalAngle importance sampling
	            const std::string&,
	            const Float&,
	            // for emitter lens
	            const tvec::TVector3<Float>&,
	            const Float&,
	            const Float&,
	            const bool&,
	            // for sensor lens
	            const tvec::TVector3<Float>&,
	            const Float&,
	            const Float&,
	            const bool&,
	            //Ultrasound parameters: a lot of them are currently not used
	#ifdef FUS_RIF
	            const Float&,
	            const Float&,
	            const Float&,
	            const Float&,
	            const int&,
	            const Float&,
	            const tvec::TVector3<Float>&,
	            const tvec::TVector3<Float>&,
	            const bool&,
	            const bool&,
	            const Float&,
	            const Float&,
	            const Float&,
	            const Float&,
	            const int&,
	            const Float&,
	            const Float&,
	            const int&,
	#else
	            const Float&,
	            const Float&,
	            const Float&,
	            const Float&,
	            const Float&,
	            const Float&,
	            const Float&,
	            const int&,
	#endif
	            const tvec::TVector3<Float>&,
	            const tvec::TVector3<Float>&,
	            const tvec::TVector3<Float>&,
	            const Float&,
	            const Float&, const Float&, const int&, const Float&, const Float&, const bool&
	#ifdef SPLINE_RIF
	//          , const Float xmin[], const Float xmax[],  const int N[]
	            , const std::string&
	#endif
            >())
            .def("set_f_u", &scn::Scene<tvec::TVector3>::set_f_u)
    #ifdef FUS_RIF
            .def("set_n_scaling", &scn::Scene<tvec::TVector3>::set_n_scaling)
            .def("set_phase1", &scn::Scene<tvec::TVector3>::set_phase1)
            .def("set_phase2", &scn::Scene<tvec::TVector3>::set_phase2)
    #else
            .def("set_n_max", &scn::Scene<tvec::TVector3>::set_n_max)
    #endif
            ;
}
