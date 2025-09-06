#pragma once
#include <constants.h>
#include <tvector.h>
#include <sampler.h>

namespace scn
{
    template <template <typename> class vector_type>
    class area_source
    {
    public:
        area_source(const vector_type<Float> &origin,
                    const vector_type<Float> &dir,
                    const tvec::Vec2f &plane,
                    Float Li);

        bool sample_ray(vector_type<Float> &pos,
                       vector_type<Float> &dir,
                       smp::Sampler &sampler) const;

        const vector_type<Float> &get_origin() const;
        const vector_type<Float> &get_dir() const;
        const tvec::Vec2f &get_plane() const;
        Float get_li() const;

        virtual ~area_source();

    protected:
        vector_type<Float> m_origin;
        vector_type<Float> m_dir;
        tvec::Vec2f m_plane;
        Float m_Li;
    };
} // namespace scn
