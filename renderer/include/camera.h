// #include <lens.h>

// #pragma once
// namespace scn{
//     template <template <typename> class vector_type>
//     class Camera
//     {
//     public:
//         Camera(const vector_type<Float> &origin,
//             const vector_type<Float> &dir,
//             const vector_type<Float> &horizontal,
//             const tvec::Vec2f &view_plane,
//             const tvec::Vec2f &pathlength_range,
//             const bool &use_bounce_decomposition,
//             const vector_type<Float> &lens_origin,
//             const Float &lens_aperture,
//             const Float &lens_focalLength,
//             const bool &lens_active) : m_origin(origin),
//                                         m_dir(dir),
//                                         m_horizontal(horizontal),
//                                         m_vertical(),
//                                         m_view_plane(view_plane),
//                                         m_pathlength_range(pathlength_range),
//                                         m_use_bounce_decomposition(use_bounce_decomposition),
//                                         m_lens(lens_origin, lens_aperture, lens_focalLength, lens_active)
//         {
//             Assert(((this->m_pathlength_range.x == -FPCONST(1.0)) && (this->m_pathlength_range.y == -FPCONST(1.0))) ||
//                 ((this->m_pathlength_range.x >= 0) && (this->m_pathlength_range.y >= this->m_pathlength_range.x)));
//             this->m_dir.normalize();
//             this->m_horizontal.normalize();
//             if (this->m_origin.dim == 3)
//             {
//                 this->m_vertical = tvec::cross(this->m_dir, this->m_horizontal);
//             }
//         }

//         /*
//         * TODO: Inline this method.
//         */
//         bool sample_position(vector_type<Float> &pos, smp::Sampler &sampler) const;

//         inline const vector_type<Float> &get_origin() const
//         {
//             return this->m_origin;
//         }

//         inline const vector_type<Float> &get_dir() const
//         {
//             return this->m_dir;
//         }

//         inline const vector_type<Float> &get_horizontal() const
//         {
//             return this->m_horizontal;
//         }

//         inline const vector_type<Float> &get_vertical() const
//         {
//             return this->m_vertical;
//         }

//         inline const tvec::Vec2f &get_plane() const
//         {
//             return this->m_view_plane;
//         }

//         inline const tvec::Vec2f &get_pathlength_range() const
//         {
//             return this->m_pathlength_range;
//         }

//         inline const bool &is_bounce_decomposition() const
//         {
//             return this->m_use_bounce_decomposition;
//         }

//         inline const Lens<vector_type> &get_lens() const
//         {
//             return this->m_lens;
//         }

//         inline const bool propagate_till_sensor(vector_type<Float> &pos, vector_type<Float> &dir, Float &total_distance) const
//         {
//             // propagate till lens
//             if (this->m_lens.is_active() && !this->m_lens.deflect(pos, dir, total_distance))
//                 return false;
//             // propagate from lens to sensor
//             Float dist = (this->m_origin[0] - pos[0]) / dir[0]; // FIXME: Assumes that the direction of propagation is in -x direction.
//             pos += dist * dir;
//             #if PRINT_DEBUGLOG
//                     if (dist < -1e-4)
//                     fmt::print(fmt::fg(fmt::color::red), "Propagation till sensor failed; dying\n");
//                     exit(EXIT_FAILURE); }
//             #endif

//             total_distance += dist;
//             return true;
//         }

//         virtual ~Camera() {}

//     protected:
//         vector_type<Float> m_origin;
//         vector_type<Float> m_dir;
//         vector_type<Float> m_horizontal;
//         vector_type<Float> m_vertical;
//         tvec::Vec2f m_view_plane;
//         tvec::Vec2f m_pathlength_range;
//         bool m_use_bounce_decomposition;
//         Lens<vector_type> m_lens;
//     };
// }




#include <lens.h>
#include <tvector.h>   // for tvec::Vec2f in the public interface
#include <sampler.h>   // for smp::Sampler

#pragma once
namespace scn {

template <template <typename> class vector_type>
class Camera
{
public:
    Camera(const vector_type<Float>& origin,
           const vector_type<Float>& dir,
           const vector_type<Float>& horizontal,
           const tvec::Vec2f&        view_plane,
           const tvec::Vec2f&        pathlength_range,
           const bool&               use_bounce_decomposition,
           const vector_type<Float>& lens_origin,
           const Float&              lens_aperture,
           const Float&              lens_focalLength,
           const bool&               lens_active);

    /*
     * TODO: Inline this method.
     */
    bool sample_position(vector_type<Float>& pos, smp::Sampler& sampler) const;

    inline const vector_type<Float>& get_origin() const        { return this->m_origin; }
    inline const vector_type<Float>& get_dir() const           { return this->m_dir; }
    inline const vector_type<Float>& get_horizontal() const     { return this->m_horizontal; }
    inline const vector_type<Float>& get_vertical() const       { return this->m_vertical; }
    inline const tvec::Vec2f&        get_plane() const         { return this->m_view_plane; }
    inline const tvec::Vec2f&        get_pathlength_range() const{ return this->m_pathlength_range; }
    inline const bool&               is_bounce_decomposition() const { return this->m_use_bounce_decomposition; }
    inline const Lens<vector_type>&  get_lens() const          { return this->m_lens; }

    // Propagate from current (pos, dir) to the sensor plane.
    // First: propagate to lens (and deflect if active). Then: propagate from lens to sensor.
    // Returns false if lens deflection rejects the ray (e.g., outside aperture).
    const bool propagate_till_sensor(vector_type<Float>& pos,
                                   vector_type<Float>& dir,
                                   Float&              total_distance) const;

    virtual ~Camera();

    Camera& operator<<(const Camera& other);

protected:
    vector_type<Float> m_origin;
    vector_type<Float> m_dir;
    vector_type<Float> m_horizontal;
    vector_type<Float> m_vertical;
    tvec::Vec2f       m_view_plane;
    tvec::Vec2f       m_pathlength_range;
    bool              m_use_bounce_decomposition;
    Lens<vector_type> m_lens;
};

} // namespace scn
