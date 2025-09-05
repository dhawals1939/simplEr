#pragma once
#include <string>
#include <sstream>
#include <cmath>


// include your real vector header here:
#include <tvector.h> // <-- replace with your actual header that defines tvec::Vec3f

#include <rif.h>
#include <json_helper.h> // <-- replace with the header that provides AnyMap/get_num/get_exact/get_num_array

class rif_sources final : public rif
{
public:
    // parameters
    Float f_u = 5 * 1e6;
    Float speed_u = 1500;
    Float n_o = 1.3333;
    Float n_scaling = 0.05e-3;
    Float n_coeff = 1;
    Float radius = 2 * 25.4e-3;

    tvec::Vec3f center1; // set in ctors
    tvec::Vec3f center2; // set in ctors

    bool active1 = true;
    bool active2 = true;

    Float phase1 = 0;
    Float phase2 = 0;

    Float chordlength = 0.5 * 25.4e-3;

    // derived
    Float theta_min = 0;
    Float theta_max = 0;
    int theta_sources = 100;

    Float trans_z_min = 0;
    Float trans_z_max = 0;
    int trans_z_sources = 501;

    // default ctor
    rif_sources()
    {
        // centers default to (-radius, 0, 0)
        this->center1 = tvec::Vec3f{-this->radius, 0.0f, 0.0f};
        this->center2 = tvec::Vec3f{-this->radius, 0.0f, 0.0f};
        this->recompute_derived_();
    }

    // ctor from AnyMap (expects the "rif_parameters" sub-map)
    explicit rif_sources(const AnyMap &p)
    {
        this->f_u = get_num<Float>(p, "f_u");
        this->speed_u = get_num<Float>(p, "speed_u");
        this->n_o = get_num<Float>(p, "n_o");
        this->n_scaling = get_num<Float>(p, "n_scaling");
        this->n_coeff = get_num<Float>(p, "n_coeff");
        this->radius = get_num<Float>(p, "radius");

        {
            auto v = get_num_array<Float>(p, "center1");
            this->center1 = tvec::Vec3f{static_cast<Float>(v.at(0)),
                                        static_cast<Float>(v.at(1)),
                                        static_cast<Float>(v.at(2))};
        }
        {
            auto v = get_num_array<Float>(p, "center2");
            this->center2 = tvec::Vec3f{static_cast<Float>(v.at(0)),
                                        static_cast<Float>(v.at(1)),
                                        static_cast<Float>(v.at(2))};
        }

        this->active1 = get_exact<bool>(p, "active1");
        this->active2 = get_exact<bool>(p, "active2");
        this->phase1 = get_num<Float>(p, "phase1");
        this->phase2 = get_num<Float>(p, "phase2");
        this->chordlength = get_num<Float>(p, "chordlength");
        this->theta_sources = get_num<int>(p, "theta_sources");
        this->trans_z_sources = get_num<int>(p, "trans_z_sources");

        this->recompute_derived_();
    }

    // ctor from raw params
    rif_sources(Float f_u_, Float speed_u_, Float n_o_,
                Float n_scaling_, Float n_coeff_, Float radius_,
                const tvec::Vec3f &center1_, const tvec::Vec3f &center2_,
                bool active1_, bool active2_,
                Float phase1_, Float phase2_,
                Float chordlength_, int theta_sources_, int trans_z_sources_)
        : f_u(f_u_), speed_u(speed_u_), n_o(n_o_), n_scaling(n_scaling_), n_coeff(n_coeff_),
          radius(radius_), center1(center1_), center2(center2_),
          active1(active1_), active2(active2_),
          phase1(phase1_), phase2(phase2_),
          chordlength(chordlength_), theta_sources(theta_sources_),
          trans_z_sources(trans_z_sources_)
    {
        this->recompute_derived_();
    }

    friend std::ostream& operator<<(std::ostream& os, const rif_sources& obj)
    {
        os << "rif_sources:\n"
           << "  f_u = " << obj.f_u << "\n"
           << "  speed_u = " << obj.speed_u << "\n"
           << "  n_o = " << obj.n_o << "\n"
           << "  n_scaling = " << obj.n_scaling << "\n"
           << "  n_coeff = " << obj.n_coeff << "\n"
           << "  radius = " << obj.radius << "\n"
           << "  center1 = (" << obj.center1[0] << ", " << obj.center1[1] << ", " << obj.center1[2] << ")\n"
           << "  center2 = (" << obj.center2[0] << ", " << obj.center2[1] << ", " << obj.center2[2] << ")\n"
           << "  active1 = " << (obj.active1 ? "true" : "false") << "\n"
           << "  active2 = " << (obj.active2 ? "true" : "false") << "\n"
           << "  phase1 = " << obj.phase1 << "\n"
           << "  phase2 = " << obj.phase2 << "\n"
           << "  chordlength = " << obj.chordlength << "\n"
           << "  theta_min = " << obj.theta_min << "\n"
           << "  theta_max = " << obj.theta_max << "\n"
           << "  theta_sources = " << obj.theta_sources << "\n"
           << "  trans_z_min = " << obj.trans_z_min << "\n"
           << "  trans_z_max = " << obj.trans_z_max << "\n"
           << "  trans_z_sources = " << obj.trans_z_sources << "\n";
        return os;
    }

private:
    void recompute_derived_()
    {
        this->theta_min = -std::asin(this->chordlength / (2 * this->radius));
        this->theta_max = std::asin(this->chordlength / (2 * this->radius));
        this->trans_z_min = -this->chordlength / 2;
        this->trans_z_max = this->chordlength / 2;
    }
};
