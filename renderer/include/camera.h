#include <lens.h>

#pragma once
namespace scn{
    template <template <typename> class vector_type>
    class Camera
    {
    public:
        Camera(const vector_type<Float> &origin,
            const vector_type<Float> &dir,
            const vector_type<Float> &horizontal,
            const tvec::Vec2f &view_plane,
            const tvec::Vec2f &pathlengthRange,
            const bool &useBounceDecomposition,
            const vector_type<Float> &lens_origin,
            const Float &lens_aperture,
            const Float &lens_focalLength,
            const bool &lens_active) : m_origin(origin),
                                        m_dir(dir),
                                        m_horizontal(horizontal),
                                        m_vertical(),
                                        m_view_plane(view_plane),
                                        m_pathlengthRange(pathlengthRange),
                                        m_useBounceDecomposition(useBounceDecomposition),
                                        m_lens(lens_origin, lens_aperture, lens_focalLength, lens_active)
        {
            Assert(((this->m_pathlengthRange.x == -FPCONST(1.0)) && (this->m_pathlengthRange.y == -FPCONST(1.0))) ||
                ((this->m_pathlengthRange.x >= 0) && (this->m_pathlengthRange.y >= this->m_pathlengthRange.x)));
            this->m_dir.normalize();
            this->m_horizontal.normalize();
            if (this->m_origin.dim == 3)
            {
                this->m_vertical = tvec::cross(this->m_dir, this->m_horizontal);
            }
        }

        /*
        * TODO: Inline this method.
        */
        bool samplePosition(vector_type<Float> &pos, smp::Sampler &sampler) const;

        inline const vector_type<Float> &get_origin() const
        {
            return this->m_origin;
        }

        inline const vector_type<Float> &get_dir() const
        {
            return this->m_dir;
        }

        inline const vector_type<Float> &getHorizontal() const
        {
            return this->m_horizontal;
        }

        inline const vector_type<Float> &getVertical() const
        {
            return this->m_vertical;
        }

        inline const tvec::Vec2f &get_plane() const
        {
            return this->m_view_plane;
        }

        inline const tvec::Vec2f &getPathlengthRange() const
        {
            return this->m_pathlengthRange;
        }

        inline const bool &isBounceDecomposition() const
        {
            return this->m_useBounceDecomposition;
        }

        inline const Lens<vector_type> &getLens() const
        {
            return this->m_lens;
        }

        inline const bool propagateTillSensor(vector_type<Float> &pos, vector_type<Float> &dir, Float &totalDistance) const
        {
            // propagate till lens
            if (this->m_lens.isActive() && !this->m_lens.deflect(pos, dir, totalDistance))
                return false;
            // propagate from lens to sensor
            Float dist = (this->m_origin[0] - pos[0]) / dir[0]; // FIXME: Assumes that the direction of propagation is in -x direction.
            pos += dist * dir;
            #if PRINT_DEBUGLOG
                    if (dist < -1e-4)
                    fmt::print(fmt::fg(fmt::color::red), "Propagation till sensor failed; dying\n");
                    exit(EXIT_FAILURE); }
            #endif

            totalDistance += dist;
            return true;
        }

        virtual ~Camera() {}

    protected:
        vector_type<Float> m_origin;
        vector_type<Float> m_dir;
        vector_type<Float> m_horizontal;
        vector_type<Float> m_vertical;
        tvec::Vec2f m_view_plane;
        tvec::Vec2f m_pathlengthRange;
        bool m_useBounceDecomposition;
        Lens<vector_type> m_lens;
    };
}