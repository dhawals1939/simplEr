#pragma once
#include <string>
#include <sstream>
#include <cmath>

#include <rif.h>
#include <json_helper.h> // AnyMap/get_num/get_exact

template <template <typename> class vector_type>
class rif_analytical final : public rif<vector_type>
{
public:
    Float m_n_max = -1;
    Float m_n_clip = -1;
    Float m_phi_min = -1;
    Float m_phi_max = -1;
    int m_mode = -1;
    
    // derived
    Float m_n_maxScaling = -1; // =n_clip/n_max

    // rif_analytical() = default;

    explicit rif_analytical(const AnyMap &rif_params, Float EgapEndLocX, Float SgapBeginLocX): rif(rif_params, EgapEndLocX, SgapBeginLocX)
    {
        this->m_f_u = get_num<Float>(rif_params, "f_u");
        this->m_speed_u = get_num<Float>(rif_params, "speed_u");
        this->m_n_max = get_num<Float>(rif_params, "n_max");
        this->m_n_clip = get_num<Float>(rif_params, "n_clip");
        this->m_phi_min = get_num<Float>(rif_params, "phi_min");
        this->m_phi_max = get_num<Float>(rif_params, "phi_max");
        this->m_mode = get_num<int>(rif_params, "mode");
        
        this->recompute_derived_();        
    }

    rif_analytical(Float f_u_, Float speed_u_, Float n_o_,
                   const vector_type<Float> &axis_uz, const vector_type<Float> &axis_ux, const vector_type<Float> &p_u,
                   Float tol, Float rrWeight, int precision, Float er_stepsize,
                   Float EgapEndLocX, Float SgapBeginLocX, bool useInitializationHack,
                   Float n_max_, Float n_clip_,
                   Float phi_min_, Float phi_max_,
                   int mode_) : rif(f_u_, speed_u_, n_o_, axis_uz, axis_ux, p_u, tol, rrWeight, precision, er_stepsize, EgapEndLocX, SgapBeginLocX, useInitializationHack), 
                   m_n_max(n_max_), m_n_clip(n_clip_), m_phi_min(phi_min_), m_phi_max(phi_max_), m_mode(mode_)
    {
        this->recompute_derived_();
    }

    friend std::ostream& operator<<(std::ostream& os, const rif_analytical& rif)
    {   
        os << static_cast<const rif&>(rif);
        os << "rif_analytical:\n"
           << "  n_max = " << rif.m_n_max << "\n"
           << "  n_clip = " << rif.m_n_clip << "\n"
           << "  phi_min = " << rif.m_phi_min << "\n"
           << "  phi_max = " << rif.m_phi_max << "\n"
           << "  mode = " << rif.m_mode << "\n";
           << "  n_maxScaling = " << rif.m_n_maxScaling << "\n";
        return os;
    }
    ~rif_analytical() override = default;

    private:
    void recompute_derived_()
    {
        assert(this->m_phi_max >= this->m_phi_min && "phi_max must be greater than or equal to phi_min");
        this->m_n_maxScaling = this->m_n_clip / this->m_n_max;
    }
};
