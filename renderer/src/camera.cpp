#include <camera.h>
#include <sampler.h>   // for smp::Sampler

#if PRINT_DEBUGLOG
  #include <fmt/color.h>
#endif

namespace scn {

template <template <typename> class vector_type>
Camera<vector_type>::Camera(const vector_type<Float>& origin,
                            const vector_type<Float>& dir,
                            const vector_type<Float>& horizontal,
                            const tvec::Vec2f&        view_plane,
                            const tvec::Vec2f&        pathlength_range,
                            const bool&               use_bounce_decomposition,
                            const vector_type<Float>& lens_origin,
                            const Float&              lens_aperture,
                            const Float&              lens_focalLength,
                            const bool&               lens_active)
    : m_origin(origin)
    , m_dir(dir)
    , m_horizontal(horizontal)
    , m_vertical()
    , m_view_plane(view_plane)
    , m_pathlength_range(pathlength_range)
    , m_use_bounce_decomposition(use_bounce_decomposition)
    , m_lens(lens_origin, lens_aperture, lens_focalLength, lens_active)
{
    Assert(((this->m_pathlength_range.x == -FPCONST(1.0)) && (this->m_pathlength_range.y == -FPCONST(1.0))) ||
           ((this->m_pathlength_range.x >= 0) && (this->m_pathlength_range.y >= this->m_pathlength_range.x)));

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
template <template <typename> class vector_type>
bool Camera<vector_type>::sample_position(vector_type<Float>& /*pos*/,
                                         smp::Sampler&       /*sampler*/) const
{
    // Stub: keep linkable until real implementation is moved here.
    return false;
}

template <template <typename> class vector_type>
const bool Camera<vector_type>::propagate_till_sensor(vector_type<Float>& pos,
                                                    vector_type<Float>& dir,
                                                    Float&              totalDistance) const
{
    // propagate till lens
    if (this->m_lens.is_active() && !this->m_lens.deflect(pos, dir, totalDistance))
        return false;

    // propagate from lens to sensor
    Float dist = (this->m_origin[0] - pos[0]) / dir[0]; // FIXME: assumes propagation in -x direction.
    pos += dist * dir;

#if PRINT_DEBUGLOG
    if (dist < -1e-4f) {
        fmt::print(fmt::fg(fmt::color::red), "Propagation till sensor failed; dying\n");
        exit(EXIT_FAILURE);
    }
#endif

    totalDistance += dist;
    return true;
}

template <template <typename> class vector_type>
Camera<vector_type>::~Camera() = default;


template <template <typename> class vector_type>
std::ostream& operator<<(std::ostream& os, const Camera<vector_type>& cam)
{
    os << "Camera(origin: " << cam.m_origin
       << ", dir: " << cam.m_dir
       << ", horizontal: " << cam.m_horizontal
       << ", vertical: " << cam.m_vertical
       << ", view_plane: " << cam.m_view_plane
       << ", pathlength_range: " << cam.m_pathlength_range
       << ", use_bounce_decomposition: " << cam.m_use_bounce_decomposition
       << ", lens: " << cam.m_lens
       << ")";
    return os;
}

// ---- Explicit template instantiations (add all you use) ----
template class Camera<tvec::TVector3>;
template class Camera<tvec::TVector2>;

} // namespace scn
