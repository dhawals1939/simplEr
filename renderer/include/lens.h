#include <tvector.h>
#include <constants.h>

#pragma once
namespace scn {

template <template <typename> class vector_type>
class Lens
{
public:
    Lens(const vector_type<Float>& origin,
         const Float&              aperture,
         const Float&              focal_length,
         const bool&               active);

    // Propagate to the lens plane located at x = origin.x.
    // Assumes propagation in the -x direction.
    // If the lens is active, it deflects the ray; otherwise, the ray passes through unchanged.
    // Returns false if the ray misses the aperture.
    const bool propagate_till_lens(vector_type<Float>& pos,
                                   vector_type<Float>& dir,
                                   Float&              total_distance) const;

    const bool               is_active()                 const;
    const vector_type<Float>& get_origin()              const;
    const Float              get_square_aperture_radius()  const;
    const Float              get_focal_length()           const;

    // Deflect the ray at the lens. Returns false if outside the aperture.
    // See the .cpp for the geometric reasoning.
    const bool deflect(const vector_type<Float>& pos,
                       vector_type<Float>&       dir,
                       Float&                    total_distance) const;

    template <template <typename> class VT>
    friend std::ostream& operator<<(std::ostream& os, const Lens<VT>& lens);
protected:
    vector_type<Float> m_origin;
    Float              m_squareApertureRadius; // squared aperture radius
    Float              m_focalLength;
    bool               m_active;               // whether the lens is present/active
};

} // namespace scn
