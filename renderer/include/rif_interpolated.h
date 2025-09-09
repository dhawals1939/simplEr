#pragma once
#include <string>
#include <sstream>

#include <rif.h>
#include <json_helper.h> // AnyMap/get_num/get_exact
template <template <typename> class vector_type>
class rif_interpolated final : public rif<vector_type>
{
public:
    std::string m_rif_grid_file = "us";

    // derived
    spline::Spline<3> m_spline;
    // rif_interpolated() = default;
    explicit rif_interpolated(const AnyMap &p, Float EgapEndLocX, Float SgapBeginLocX): rif(p, EgapEndLocX, SgapBeginLocX)
    {
        this->m_rif_grid_file = get_exact<std::string>(p, "rif_grid_file");
        
        this->recompute_derived_();
    }

    rif_interpolated(Float f_u_, Float speed_u_, Float n_o_,
                     const vector_type<Float> &axis_uz, const vector_type<Float> &axis_ux, const vector_type<Float> &p_u,
                     Float tol, Float rrWeight, int precision, Float er_stepsize,
                     Float EgapEndLocX, Float SgapBeginLocX, bool useInitializationHack, 
                     std::string rifgrid_file_)
        : rif(f_u_, speed_u_, n_o_, axis_uz, axis_ux, p_u, tol, rrWeight, precision, er_stepsize, EgapEndLocX, SgapBeginLocX, useInitializationHack), m_rif_grid_file(std::move(rifgrid_file_))
    {
        this->recompute_derived_();
    }

    friend std::ostream& operator<<(std::ostream& os, const rif_interpolated& obj)
    {   
        os << static_cast<const rif&>(obj);
        os << "rif_interpolated:\n"
           << "  rif_grid_file = " << obj.m_rif_grid_file << "\n";
        return os;
    }

    ~rif_interpolated() override = default;

protected:
    // ----- Required overrides: inside-gap math only -----

    Float compute_refractive_index(const Vec &p, Float scaling) const override
    {
        /* // Old code for 2D spline
              Float temp[2];
              temp[0] = p.y;
              temp[1] = p.z;
              return (m_spline.value<0, 0>(temp)*scaling + n_o);
        */

        Float temp[3];
        temp[0] = p.x;
        temp[1] = p.y;
        temp[2] = p.z;
        return this->m_spline.value(temp);
    }

    Vec compute_refractive_index_gradient(const Vec &p, Float scaling) const override
    {
        /* // Old code for 2D spline
                vector_type<Float>(m_spline.value<1, 0, 0>(temp), m_spline.value<0, 1, 0>(temp), m_spline.value<0, 0, 1>(temp));
                Float temp[2];
                temp[0] = q.z;
                temp[1] = q.y;
        
                // return scaling*m_spline.gradient2d(temp);
                return scaling*vector_type<Float>(0.0, m_spline.value<0, 1>(temp), m_spline.value<1, 0>(temp));
        */
        Float temp[3];
        temp[0] = p.x;
        temp[1] = p.y;
        temp[2] = p.z;
        return this->m_spline.gradient(temp);
    }

    Mat3 compute_refractive_index_hessian(const Vec &p, Float scaling) const override
    {
        /* // Old code for 2D spline
                 Float temp[2];
                 temp[0] = p.z;
                 temp[1] = p.y;
                 return scaling*m_spline.hessian2d(temp);
                 Float hxy = m_spline.value<1, 1>(temp);
                 return scaling*Matrix3x3(0, 0,   0,
                                   0, m_spline.value<0, 2>(temp), hxy,
                                   0, hxy,                        m_spline.value<2, 0>(temp));
        */
        Float temp[3];
        temp[0] = p.x;
        temp[1] = p.y;
        temp[2] = p.z;
        return this->m_spline.hessian(temp);
       
    }

private:
    void recompute_derived_()
    {
        this->spline = spline::Spline<3>(this->m_rif_grid_file);
    }
};
