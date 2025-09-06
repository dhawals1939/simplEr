#pragma once
#include <string>
#include <sstream>
#include <cmath>

#include <rif.h>
#include <json_helper.h> // AnyMap/get_num/get_exact


class rif_analytical final : public rif
{
public:
    Float m_f_u = 848 * 1e3;
    Float m_speed_u = 1500;
    Float m_n_o = 1.3333;
    Float m_n_max = 1e-3;
    Float m_n_clip = 1e-3;
    Float m_phi_min = M_PI / 2;
    Float m_phi_max = M_PI / 2;
    int m_mode = 0;

    rif_analytical() = default;

    explicit rif_analytical(const AnyMap &rif_params)
    {
        this->m_f_u = get_num<Float>(rif_params, "f_u");
        this->m_speed_u = get_num<Float>(rif_params, "speed_u");
        this->m_n_o = get_num<Float>(rif_params, "n_o");
        this->m_n_max = get_num<Float>(rif_params, "n_max");
        this->m_n_clip = get_num<Float>(rif_params, "n_clip");
        this->m_phi_min = get_num<Float>(rif_params, "phi_min");
        this->m_phi_max = get_num<Float>(rif_params, "phi_max");
        this->m_mode = get_num<int>(rif_params, "mode");
        assert(this->m_phi_max >= this->m_phi_min && "phi_max must be greater than or equal to phi_min");
    }

    rif_analytical(Float f_u_, Float speed_u_, Float n_o_,
                 Float n_max_, Float n_clip_,
                 Float phi_min_, Float phi_max_,
                 int mode_)
    {
        this->m_f_u = f_u_;
        this->m_speed_u = speed_u_;
        this->m_n_o = n_o_;
        this->m_n_max = n_max_;
        this->m_n_clip = n_clip_;
        this->m_phi_min = phi_min_;
        this->m_phi_max = phi_max_;
        this->m_mode = mode_;
    }

    friend std::ostream& operator<<(std::ostream& os, const rif_analytical& rif)
    {
        os << "rif_analytical:\n"
           << "  f_u = " << rif.m_f_u << "\n"
           << "  speed_u = " << rif.m_speed_u << "\n"
           << "  n_o = " << rif.m_n_o << "\n"
           << "  n_max = " << rif.m_n_max << "\n"
           << "  n_clip = " << rif.m_n_clip << "\n"
           << "  phi_min = " << rif.m_phi_min << "\n"
           << "  phi_max = " << rif.m_phi_max << "\n"
           << "  mode = " << rif.m_mode << "\n";
        return os;
    }
};
