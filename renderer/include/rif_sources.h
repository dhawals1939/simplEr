#pragma once
#include <string>
#include <sstream>
#include <cmath>


// include your real vector header here:
#include <tvector.h> // <-- replace with your actual header that defines tvec::Vec3f

#include <rif.h>
#include <json_helper.h> // <-- replace with the header that provides AnyMap/get_num/get_exact/get_num_array

template <template <typename> class vector_type>
class rif_sources final : public rif<vector_type>
{
public:
    // parameters
    Float m_n_scaling = -1;
    Float m_n_coeff = -1;
    Float m_radius = -1;

    tvec::Vec3f m_center1; // set in ctors
    tvec::Vec3f m_center2; // set in ctors

    bool m_active1 = false;
    bool m_active2 = false;

    Float m_phase1 = -1;
    Float m_phase2 = -1;

    Float m_chordlength = -1;

    // derived
    Float m_theta_min = -1;
    Float m_theta_max = -1;
    int m_theta_sources = -1;

    Float m_trans_z_min = -1;
    Float m_trans_z_max = -1;
    int m_trans_z_sources = -1;

    int m_nsources = -1;

    vector_type<Float> *m_centers1;
    vector_type<Float> *m_centers2;

    // // default ctor
    // rif_sources()
    // {
    //     // centers default to (-radius, 0, 0)
    //     this->m_center1 = tvec::Vec3f{-this->m_radius, 0.0f, 0.0f};
    //     this->m_center2 = tvec::Vec3f{-this->m_radius, 0.0f, 0.0f};
    //     this->recompute_derived_();
    // }
    

    // ctor from AnyMap (expects the "rif_parameters" sub-map)
    explicit rif_sources(const AnyMap &p, const Float EgapEndLocX, const Float SgapBeginLocX): rif(p, EgapEndLocX, SgapBeginLocX)  // Initialize base class from AnyMap
    {
        this->m_n_scaling = get_num<Float>(p, "n_scaling");
        this->m_n_coeff = get_num<Float>(p, "n_coeff");
        this->m_radius = get_num<Float>(p, "radius");

        {
            auto v = get_num_array<Float>(p, "center1");
            this->m_center1 = tvec::Vec3f{static_cast<Float>(v.at(0)),
                                        static_cast<Float>(v.at(1)),
                                        static_cast<Float>(v.at(2))};
        }
        {
            auto v = get_num_array<Float>(p, "center2");
            this->m_center2 = tvec::Vec3f{static_cast<Float>(v.at(0)),
                                        static_cast<Float>(v.at(1)),
                                        static_cast<Float>(v.at(2))};
        }

        this->m_active1 = get_exact<bool>(p, "active1");
        this->m_active2 = get_exact<bool>(p, "active2");
        this->m_phase1 = get_num<Float>(p, "phase1");
        this->m_phase2 = get_num<Float>(p, "phase2");
        this->m_chordlength = get_num<Float>(p, "chordlength");
        this->m_theta_sources = get_num<int>(p, "theta_sources");
        this->m_trans_z_sources = get_num<int>(p, "trans_z_sources");
        this->recompute_derived_();
    }

    // ctor from raw params
    rif_sources(Float f_u_, Float speed_u_, Float n_o_,
                const vector_type<Float> &axis_uz, const vector_type<Float> &axis_ux, const vector_type<Float> &p_u,
                Float tol, Float rrWeight, int precision, Float er_stepsize,
                Float EgapEndLocX, Float SgapBeginLocX, bool useInitializationHack,
                Float n_scaling_, Float n_coeff_, Float radius_,
                const tvec::Vec3f &center1_, const tvec::Vec3f &center2_,
                bool active1_, bool active2_,
                Float phase1_, Float phase2_,
                Float chordlength_, int theta_sources_, int trans_z_sources_)
        : rif(f_u_, speed_u_, n_o_, axis_uz, axis_ux, p_u, tol, rrWeight, precision, er_stepsize, EgapEndLocX, SgapBeginLocX, useInitializationHack),
          m_n_scaling(n_scaling_), m_n_coeff(n_coeff_),
          m_radius(radius_), m_center1(center1_), m_center2(center2_),
          m_active1(active1_), m_active2(active2_),
          m_phase1(phase1_), m_phase2(phase2_),
          m_chordlength(chordlength_), m_theta_sources(theta_sources_),
          m_trans_z_sources(trans_z_sources_)
    {
        this->recompute_derived_();
    }

    friend std::ostream& operator<<(std::ostream& os, const rif_sources& obj)
    {
        os << static_cast<const rif&>(obj);
        os << "rif_sources:\n"
           << "  n_scaling = " << obj.m_n_scaling << "\n"
           << "  n_coeff = " << obj.m_n_coeff << "\n"
           << "  radius = " << obj.m_radius << "\n"
           << "  center1 = (" << obj.m_center1[0] << ", " << obj.m_center1[1] << ", " << obj.m_center1[2] << ")\n"
           << "  center2 = (" << obj.m_center2[0] << ", " << obj.m_center2[1] << ", " << obj.m_center2[2] << ")\n"
           << "  active1 = " << (obj.m_active1 ? "true" : "false") << "\n"
           << "  active2 = " << (obj.m_active2 ? "true" : "false") << "\n"
           << "  phase1 = " << obj.m_phase1 << "\n"
           << "  phase2 = " << obj.m_phase2 << "\n"
           << "  chordlength = " << obj.m_chordlength << "\n"
           << "  theta_min = " << obj.m_theta_min << "\n"
           << "  theta_max = " << obj.m_theta_max << "\n"
           << "  theta_sources = " << obj.m_theta_sources << "\n"
           << "  trans_z_min = " << obj.m_trans_z_min << "\n"
           << "  trans_z_max = " << obj.m_trans_z_max << "\n"
           << "  trans_z_sources = " << obj.m_trans_z_sources << "\n";
           << "  nsources = " << obj.m_nsources << "\n";
        return os;
    }

    ~rif_sources() noexcept override
    {
        if (this->m_active1 && this->m_centers1)
            delete[] this->m_centers1;
        if (this->m_active2 && this->m_centers2)
            delete[] this->m_centers2;
    }

protected:
    // ----- Required overrides: inside-gap math only -----

    Float compute_refractive_index(const Vec &p, Float scaling) const override
    {
    }

    Vec compute_refractive_index_gradient(const Vec &p, Float scaling) const override
    {
    }

    Mat3 compute_refractive_index_hessian(const Vec &p, Float scaling) const override
    {
    }

private:
    void recompute_derived_()
    {
        this->m_theta_min = -std::asin(this->m_chordlength / (2 * this->m_radius));
        this->m_theta_max = std::asin(this->m_chordlength / (2 * this->m_radius));
        this->m_trans_z_min = -this->m_chordlength / 2;
        this->m_trans_z_max = this->m_chordlength / 2;
        this->nsources = this->m_theta_sources * this->m_trans_z_sources;

        Float theta_diff = (m_theta_max - m_theta_min) / (m_theta_sources - 1);
        Float trans_z_diff = (m_trans_z_max - m_trans_z_min) / (m_trans_z_sources - 1);

        if (this->m_active1)
            this->m_centers1 = new vector_type<Float>[this->nsources];
        if (this->m_active2)
            this->m_centers2 = new vector_type<Float>[this->nsources];

        for (int i = 0; i < this->m_theta_sources; i++)
        {
            Float theta = this->m_theta_min + theta_diff * i;
            Float xval = this->m_radius * (1 - cos(theta));
            Float yval = this->m_radius * (sin(theta));

            for (int j = 0; j < this->m_trans_z_sources; j++)
            {
                // int index = i*trans_z_sources + j;
                int index = i + j * this->m_theta_sources; // to match matlab indices and debug
                // for horizontal (0, 0, 0.0508)
                if (this->m_active1)
                {
                    this->m_centers1[index].y = yval + this->m_center1.y;
                    this->m_centers1[index].z = xval + this->m_center1.z;
                    this->m_centers1[index].x = this->m_trans_z_min + trans_z_diff * j + this->m_center1.x;
                }
                // for vertical (0, -0.0508, 0)
                if (this->m_active2)
                {
                    this->m_centers2[index].y = xval + this->m_center2.y;
                    this->m_centers2[index].z = yval + this->m_center2.z;
                    this->m_centers2[index].x = this->m_trans_z_min + trans_z_diff * j + this->m_center2.x;
                }
            }
        }
    }
};
