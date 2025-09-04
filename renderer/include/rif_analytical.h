#pragma once
#include <string>
#include <sstream>
#include <cmath>

#include <rif.h>
#include <json_helper.h> // AnyMap/get_num/get_exact


class rif_analytical final : public rif
{
public:
    Float f_u = 848 * 1e3;
    Float speed_u = 1500;
    Float n_o = 1.3333;
    Float n_max = 1e-3;
    Float n_clip = 1e-3;
    Float phi_min = M_PI / 2;
    Float phi_max = M_PI / 2;
    int mode = 0;

    rif_analytical() = default;

    explicit rif_analytical(const AnyMap &rif_params)
    {
        this->f_u = get_num<Float>(rif_params, "f_u");
        this->speed_u = get_num<Float>(rif_params, "speed_u");
        this->n_o = get_num<Float>(rif_params, "n_o");
        this->n_max = get_num<Float>(rif_params, "n_max");
        this->n_clip = get_num<Float>(rif_params, "n_clip");
        this->phi_min = get_num<Float>(rif_params, "phi_min");
        this->phi_max = get_num<Float>(rif_params, "phi_max");
        this->mode = get_num<int>(rif_params, "mode");
        assert(this->phi_max >= this->phi_min && "phi_max must be greater than or equal to phi_min");
    }

    rif_analytical(Float f_u_, Float speed_u_, Float n_o_,
                 Float n_max_, Float n_clip_,
                 Float phi_min_, Float phi_max_,
                 int mode_)
    {
        this->f_u = f_u_;
        this->speed_u = speed_u_;
        this->n_o = n_o_;
        this->n_max = n_max_;
        this->n_clip = n_clip_;
        this->phi_min = phi_min_;
        this->phi_max = phi_max_;
        this->mode = mode_;
    }

    std::string to_string() const override
    {
        std::ostringstream oss;
        oss << "rif_analytical:\n";
        oss << "  f_u = " << this->f_u << "\n";
        oss << "  speed_u = " << this->speed_u << "\n";
        oss << "  n_o = " << this->n_o << "\n";
        oss << "  n_max = " << this->n_max << "\n";
        oss << "  n_clip = " << this->n_clip << "\n";
        oss << "  phi_min = " << this->phi_min << "\n";
        oss << "  phi_max = " << this->phi_max << "\n";
        oss << "  mode = " << this->mode << "\n";
        return oss.str();
    }
};
