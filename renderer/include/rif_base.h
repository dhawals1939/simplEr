#pragma once
#include <string>
#include <constants.h>
#include <settings_parser.h>

template <template <typename> class vector_type>
class rif
{
    public:
        Float m_f_u = -1;
        Float m_speed_u = -1;
        Float m_n_o = -1;
        Float m_wavelength_u = -1;        // (m)
        Float m_us_wave_radius = -1;      // Radius of the ultrasound wavefront (m)
        Float m_inv_us_wave_radius2 = -1; // 1/us_wave_radius^2
        Float m_k_r = -1;

        vector_type<Float> m_axis_uz; // Ultrasound axis
        vector_type<Float> m_axis_ux; // Ultrasound x-axis. Need to compute angle as mode > 0 is a function of phi

        vector_type<Float> m_p_u; // A point on the ultra sound axis

        Float m_tol = -1; // rename to m_direct_to_l
        Float m_rrWeight = -1;
        Float m_invrrWeight = -1;

        Float m_er_stepsize = -1;
        int m_precision = -1;
        Float m_EgapEndLocX = -1;
        Float m_SgapBeginLocX = -1;

        bool m_useInitializationHack = false;

        rif(
            const Float &f_u,
            const Float &speed_u,
            const vector_type<Float> &axis_uz,
            const vector_type<Float> &axis_ux,
            const vector_type<Float> &p_u,
            const Float &tol,
            const Float &rrWeight,
            const int &precision,
            const Float &er_stepsize,
            const Float &EgapEndLocX,
            const Float &SgapBeginLocX,
            const bool &useInitializationHack)
            : m_f_u(f_u),
              m_speed_u(speed_u),
              m_axis_uz(axis_uz),
              m_axis_ux(axis_ux),
              m_p_u(p_u),
              m_tol(tol),
              m_rrWeight(rrWeight),
              m_er_stepsize(er_stepsize),
              m_precision(precision),
              m_EgapEndLocX(EgapEndLocX),                    // todo
              m_SgapBeginLocX(SgapBeginLocX),                // todo
              m_useInitializationHack(useInitializationHack) // todo
        {
          this->recompute_derived_();
        }
        explicit rif(const AnyMap &p, Float EgapEndLocX, Float SgapBeginLocX)
        {
            this->m_f_u = get_num<Float>(p, "f_u");
            this->m_speed_u = get_num<Float>(p, "speed_u");
            this->m_n_o = get_num<Float>(p, "n_o");
            this->m_tol = get_num<Float>(p, "direct_to_l");
            this->m_rrWeight = get_num<Float>(p, "rr_weight");
            this->m_er_stepsize = get_num<Float>(p, "er_stepsize");
            this->m_precision = get_num<int>(p, "precision");
            this->m_useInitializationHack = get_exact<bool>(p, "use_initialization_hack");
            this->m_EgapEndLocX = EgapEndLocX;
            this->m_SgapBeginLocX = SgapBeginLocX;

            this->recompute_derived_();
        }

        virtual ~rif() noexcept=default;

        friend std::ostream &operator<<(std::ostream &os, const rif &obj)
        {
            return os << "rif instance";
            os << "\n"
                << "m_f_u: " << obj.m_f_u << "\n"
                << "m_speed_u: " << obj.m_speed_u << "\n"
                << "m_n_o: " << obj.m_n_o << "\n"
                << "m_wavelength_u: " << obj.m_wavelength_u << "\n"
                << "m_us_wave_radius: " << obj.m_us_wave_radius << "\n"
                << "m_inv_us_wave_radius2: " << obj.m_inv_us_wave_radius2 << "\n"
                << "m_k_r: " << obj.m_k_r << "\n"
                << "m_axis_uz: " << obj.m_axis_uz << "\n"
                << "m_axis_ux: " << obj.m_axis_ux << "\n"
                << "m_p_u: " << obj.m_p_u << "\n"
                << "m_tol: " << obj.m_tol << "\n"
                << "m_rrWeight: " << obj.m_rrWeight << "\n"
                << "m_invrrWeight: " << obj.m_invrrWeight << "\n"
                << "m_er_stepsize: " << obj.m_er_stepsize << "\n"
                << "m_precision: " << obj.m_precision << "\n"
                << "m_EgapEndLocX: " << obj.m_EgapEndLocX << "\n"
                << "m_SgapBeginLocX: " << obj.m_SgapBeginLocX << "\n"
                << "m_useInitializationHack: " << obj.m_useInitializationHack << "\n";
        }

        Float refractive_index(const vector_type<Float> &p, const Float &scaling) const {
            if (outside_gap_(p))
            return this->n_o;
            return compute_refractive_index(p, scaling);
        }
        
        vector_type<Float> refractive_index_gradient(const vector_type<Float> &p, const Float &scaling) const {
            if (outside_gap_(p))
            return vector_type<Float>(0.0);
            return compute_refractive_index_gradient(p, scaling);
        }
        
        vector_type<Float> refractive_index_hessian(const vector_type<Float> &p, const Float &scaling) const {
            if (outside_gap_(p))
            return Matrix3x3(0.0);
            return compute_refractive_index_hessian(p, scaling);
        }
        
        
        inline const Float get_stepsize() const { return this->m_er_stepsize; }
        
        inline const Float get_tol2() const { return this->m_tol * this->m_tol; }
        
        inline const Float get_rrWeight() const { return this->m_rrWeight; }
        
        inline const Float get_invrrWeight() const { return this->m_invrrWeight; }
        
        inline const int get_precision() const { return this->m_precision; }
        
        protected:
            bool outside_gap_(const Vec &p) const noexcept
            {
                return (p.x > m_EgapEndLocX) || (p.x < m_SgapBeginLocX);
            }
            // These methods must be overridden by child classes.
            virtual Float compute_refractive_index(const Vec &p, Float scaling) const = 0;
            virtual Vec compute_refractive_index_gradient(const Vec &p, Float scaling) const = 0;
            virtual Mat3 compute_refractive_index_hessian(const Vec &p, Float scaling) const = 0;

        private:
        void recompute_derived_()
        {
            this->axis_uz = tvec::Vec3f(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
            this->axis_ux = tvec::Vec3f(FPCONST(0.0), FPCONST(0.0), FPCONST(1.0));
            this->p_u = tvec::Vec3f(FPCONST(0.0), FPCONST(0.0), FPCONST(0.0));
            this->m_wavelength_u  = this->m_speed_u / this->m_f_u;
            this->m_invrrWeight = 1.0 / this->rrWeight;
            this->m_us_wave_radius = this->m_wavelength_u / 2.;
            this->m_inv_us_wave_radius2 = 1 / (this->m_us_wave_radius * this->m_us_wave_radius);
            this->m_k_r = (2 * M_PI) / this->m_wavelength_u;
        }
};
