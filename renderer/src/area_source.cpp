#include <area_source.h>

#if USE_PRINTING
#include <fmt/core.h>
#endif

namespace scn
{
    template <template <typename> class VectorType>
    area_source<VectorType>::area_source(const VectorType<Float> &origin,
                                         const VectorType<Float> &dir,
                                         const tvec::Vec2f &plane,
                                         Float Li)
        : m_origin(origin),
          m_dir(dir),
          m_plane(plane),
          m_Li(Li)
    {
        this->m_dir.normalize();
#if USE_PRINTING
        fmt::print(" dir {} {}", this->m_dir.x, this->m_dir.y);
        if (this->m_dir.dim == 3)
            fmt::print(" {}\n", this->m_dir.z);
        else
            fmt::print("\n");
#endif
    }

    template <template <typename> class VectorType>
    bool area_source<VectorType>::sample_ray(VectorType<Float> &pos,
                                            VectorType<Float> &dir,
                                            smp::Sampler &sampler) const
    {
        // TODO: real implementation
        return false;
    }

    template <template <typename> class VectorType>
    const VectorType<Float> &area_source<VectorType>::get_origin() const
    {
        return this->m_origin;
    }

    template <template <typename> class VectorType>
    const VectorType<Float> &area_source<VectorType>::get_dir() const
    {
        return this->m_dir;
    }

    template <template <typename> class VectorType>
    const tvec::Vec2f &area_source<VectorType>::get_plane() const
    {
        return this->m_plane;
    }

    template <template <typename> class VectorType>
    Float area_source<VectorType>::get_Li() const
    {
        return this->m_Li;
    }

    template <template <typename> class VectorType>
    area_source<VectorType>::~area_source() = default;

    // ---- Explicit instantiations (add all VectorType templates you use) ----
    template class area_source<tvec::TVector3>;
    template class area_source<tvec::TVector2>;
} // namespace scn
