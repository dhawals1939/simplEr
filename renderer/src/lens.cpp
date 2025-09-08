#include <lens.h>
// #include <cassert>  // If Assert is not your project macro, replace Assert(...) with assert(...)

namespace scn {

template <template <typename> class vector_type>
Lens<vector_type>::Lens(const vector_type<Float>& origin,
                        const Float&              aperture,
                        const Float&              focal_length,
                        const bool&               active)
    : m_origin(origin)
    , m_squareApertureRadius(aperture * aperture)
    , m_focalLength(focal_length)
    , m_active(active)
{}

// Propagate to the lens plane at x = origin.x (assumes -x propagation). If active,
// call deflect; otherwise just report success.
template <template <typename> class vector_type>
const bool Lens<vector_type>::propagate_till_lens(vector_type<Float>& pos,
                                                  vector_type<Float>& dir,
                                                  Float&              total_distance) const
{
    // Distance along -x to reach the lens plane (x = m_origin.x).
    Float dist = -(pos[0] - this->m_origin[0]) / dir[0]; // NOTE: assumes -x propagation.
    pos += dist * dir;
    total_distance += dist;

    if (this->m_active)
        return this->deflect(pos, dir, total_distance);
    else
        return true;
}

template <template <typename> class vector_type>
const bool Lens<vector_type>::is_active() const
{
    return this->m_active;
}

template <template <typename> class vector_type>
const vector_type<Float>& Lens<vector_type>::get_origin() const
{
    return this->m_origin;
}

template <template <typename> class vector_type>
const Float Lens<vector_type>::get_square_aperture_radius() const
{
    return this->m_squareApertureRadius;
}

template <template <typename> class vector_type>
const Float Lens<vector_type>::get_focal_length() const
{
    return this->m_focalLength;
}

/*
 * Deflection computation:
 * A point traveling through the center of the lens and parallel to the incident direction is [pos.x, 0, 0].
 * A ray from that point (through the optical center) goes straight and hits the focal plane at:
 *     (pos.x - f, -d_y * f / d_x, -d_z * f / d_x)   assuming propagation toward -x.
 * The original ray is deflected to pass through this focal point.
 * The (negative) additional path length at the lens is:
 *     -f/d_x - || focal_point - original_point ||
 * Returns false if the hit point is outside the (squared) aperture radius.
 */
template <template <typename> class vector_type>
const bool Lens<vector_type>::deflect(const vector_type<Float>& pos,
                                      vector_type<Float>&       dir,
                                      Float&                    total_distance) const
{
    // Check aperture: compute squared distance from the lens origin in the transverse plane (y/z or y in 2D).
    Float squareDistFromLensOrigin = 0.0f;
    for (int i = 1; i < pos.dim; ++i)
        squareDistFromLensOrigin += pos[i] * pos[i];
    if (squareDistFromLensOrigin > this->m_squareApertureRadius)
        return false;

    // Ensure we are on the lens plane (project’s Assert macro).
    Assert(pos.x == this->m_origin.x);

    // Compute deflection toward the focal plane.
    Float invd = -1 / dir[0];              // 1 / (-d_x)
    dir[0] = -this->m_focalLength;
    dir[1] = dir[1] * invd * this->m_focalLength - pos[1];
    dir[2] = dir[2] * invd * this->m_focalLength - pos[2];

    // Update total distance: -f/d_x minus length of the new direction before normalization.
    total_distance += this->m_focalLength * invd - dir.length();

    // Normalize direction.
    dir.normalize();

    return true;
}


template <template <typename> class vector_type>
std::ostream& operator<<(std::ostream& os, const Lens<vector_type>& lens)
{
    os << "Lens(origin=" << lens.get_origin()
       << ", aperture^2=" << lens.get_square_aperture_radius()
       << ", focal_length=" << lens.get_focal_length()
       << ", active=" << lens.is_active() << ")";
    return os;
}

// -------- Explicit template instantiations --------
template class Lens<tvec::TVector3>;
template class Lens<tvec::TVector2>;

} // namespace scn
