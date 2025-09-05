// #pragma once
// namespace scn
// {
//     template <template <typename> class vector_type>
//     class area_source
//     {
//     public:
//         area_source(const vector_type<Float> &origin, const vector_type<Float> &dir,
//                 const tvec::Vec2f &plane, Float Li)
//             : m_origin(origin),
//             m_dir(dir),
//             m_plane(plane),
//             m_Li(Li)
//         { /* m_dir(std::cos(angle), std::sin(angle), FPCONST(0.0)) */
//             /*
//             * TODO: Added this check for 2D version.
//             */
//             this->m_dir.normalize();
//     #if USE_PRINTING
//             fmt::print(" dir {} {}", this->m_dir.x, this->m_dir.y);
//             if (this->m_dir.dim == 3)
//             {
//                 fmt::print(" {}\n", this->m_dir.z);
//             }
//             else
//             {
//                 fmt::print("\n");
//             }
//     #endif
//         }

//         /*
//         * TODO: Inline this method.
//         */
//         bool sample_ray(vector_type<Float> &pos, vector_type<Float> &dir, smp::Sampler &sampler) const;

//         inline const vector_type<Float> &get_origin() const
//         {
//             return this->m_origin;
//         }

//         inline const vector_type<Float> &get_dir() const
//         {
//             return this->m_dir;
//         }

//         inline const tvec::Vec2f &get_plane() const
//         {
//             return this->m_plane;
//         }

//         inline Float get_li() const
//         {
//             return this->m_Li;
//         }

//         virtual ~area_source() {}

//     protected:
//         vector_type<Float> m_origin;
//         vector_type<Float> m_dir;
//         tvec::Vec2f m_plane;
//         Float m_Li;
//     };
// }

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
