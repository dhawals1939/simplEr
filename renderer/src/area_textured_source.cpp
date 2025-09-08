#include <area_textured_source.h>

namespace scn
{
    template <template <typename> class vector_type>
    area_textured_source<vector_type>::area_textured_source(
        const vector_type<Float>& origin,
        const vector_type<Float>& dir,
        const Float& half_theta_limit,
        const std::string& filename,
        const tvec::Vec2f& plane,
        const Float& Li,
        const vector_type<Float>& lens_origin,
        const Float& lens_aperture,
        const Float& lens_focal_length,
        const bool&  lens_active,
        const EmitterType& emittertype)
        : m_origin(origin),
          m_dir(dir),
          m_half_theta_limit(half_theta_limit),
          m_ct(0),
          m_texture(),
          m_texture_sampler(),
          m_pixelsize(),
          m_plane(plane),
          m_Li(Li),
          m_emittertype(emittertype),
          m_lens(lens_origin, lens_aperture, lens_focal_length, lens_active)
    {
        this->m_texture.read_file(filename);

        const int xres = this->m_texture.get_x_res();
        const int yres = this->m_texture.get_y_res();
        const int length = xres * yres;

        this->m_pixelsize.x = this->m_plane.x / xres;
        this->m_pixelsize.y = this->m_plane.y / yres;

        this->m_ct = std::cos(this->m_half_theta_limit);

        this->m_texture_sampler.reserve(length);
        for (int i = 0; i < length; ++i) {
            this->m_texture_sampler.append(this->m_texture.get_pixel(i));
        }
        this->m_texture_sampler.normalize();
    }

    template <template <typename> class vector_type>
    bool area_textured_source<vector_type>::sample_ray(
        vector_type<Float>& pos,
        vector_type<Float>& dir,
        smp::Sampler& sampler,
        Float& total_distance) const
    {
        // TODO: provide real implementation
        (void)pos; (void)dir; (void)sampler; (void)total_distance;
        return false;
    }

    template <template <typename> class vector_type>
    const vector_type<Float>& area_textured_source<vector_type>::get_origin() const {
        return this->m_origin;
    }

    template <template <typename> class vector_type>
    const vector_type<Float>& area_textured_source<vector_type>::get_dir() const {
        return this->m_dir;
    }

    template <template <typename> class vector_type>
    const tvec::Vec2f& area_textured_source<vector_type>::get_plane() const {
        return this->m_plane;
    }

    template <template <typename> class vector_type>
    const Float area_textured_source<vector_type>::get_half_theta_limit() const {
        return this->m_half_theta_limit;
    }

    template <template <typename> class vector_type>
    const typename area_textured_source<vector_type>::EmitterType&
    area_textured_source<vector_type>::get_emitter_type() const {
        return this->m_emittertype;
    }

    template <template <typename> class vector_type>
    const image::Texture& area_textured_source<vector_type>::get_texture() const {
        return this->m_texture;
    }

    template <template <typename> class vector_type>
    const Float area_textured_source<vector_type>::get_Li() const {
        return this->m_Li;
    }

    template <template <typename> class vector_type>
    const Lens<vector_type>& area_textured_source<vector_type>::get_lens() const {
        return this->m_lens;
    }

    template <template <typename> class vector_type>
    const std::vector<Float>&
    area_textured_source<vector_type>::texture_sampler_cdf() const {
        return this->m_texture_sampler.get_cdf();
    }

    template <template <typename> class vector_type>
    const bool area_textured_source<vector_type>::propagate_till_medium(
        vector_type<Float>& pos,
        vector_type<Float>& dir,
        Float& total_distance) const
    {
        if (!this->m_lens.propagate_till_lens(pos, dir, total_distance))
            return false;
        return true;
    }

    template <template <typename> class vector_type>
    area_textured_source<vector_type>::~area_textured_source() = default;


    template <template <typename> class vector_type>
    std::ostream& operator<<(std::ostream& os, const area_textured_source<vector_type>& src) {
        os << "area_textured_source {\n";
        os << "  origin: " << src.get_origin() << "\n";
        os << "  dir: " << src.get_dir() << "\n";
        os << "  half_theta_limit: " << src.get_half_theta_limit() << "\n";
        os << "  plane: " << src.get_plane() << "\n";
        os << "  Li: " << src.get_Li() << "\n";
        os << "  emittertype: " << src.get_emitter_type() << "\n";
        os << "  lens: " << src.get_lens() << "\n";
        os << "  texture: [xres=" << src.get_texture().get_x_res()
           << ", yres=" << src.get_texture().get_y_res() << "]\n";
        os << "}";
        return os;
    }

    // ---- Explicit instantiations (adjust to what you actually use) ----
    template class area_textured_source<tvec::TVector3>;
    template class area_textured_source<tvec::TVector2>;
} // namespace scn
