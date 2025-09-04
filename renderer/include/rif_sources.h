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

    std::string to_string() const override
    {
        std::ostringstream oss;
        oss << "rif_sources:\n";
        oss << "  f_u = " << this->f_u << "\n";
        oss << "  speed_u = " << this->speed_u << "\n";
        oss << "  n_o = " << this->n_o << "\n";
        oss << "  n_scaling = " << this->n_scaling << "\n";
        oss << "  n_coeff = " << this->n_coeff << "\n";
        oss << "  radius = " << this->radius << "\n";
        oss << "  center1 = (" << this->center1[0] << ", " << this->center1[1] << ", " << this->center1[2] << ")\n";
        oss << "  center2 = (" << this->center2[0] << ", " << this->center2[1] << ", " << this->center2[2] << ")\n";
        oss << "  active1 = " << (this->active1 ? "true" : "false") << "\n";
        oss << "  active2 = " << (this->active2 ? "true" : "false") << "\n";
        oss << "  phase1 = " << this->phase1 << "\n";
        oss << "  phase2 = " << this->phase2 << "\n";
        oss << "  chordlength = " << this->chordlength << "\n";
        oss << "  theta_min = " << this->theta_min << "\n";
        oss << "  theta_max = " << this->theta_max << "\n";
        oss << "  theta_sources = " << this->theta_sources << "\n";
        oss << "  trans_z_min = " << this->trans_z_min << "\n";
        oss << "  trans_z_max = " << this->trans_z_max << "\n";
        oss << "  trans_z_sources = " << this->trans_z_sources << "\n";
        return oss.str();
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
