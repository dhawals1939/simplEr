#pragma once

namespace scn
{
    template <template <typename> class vector_type>
    class area_textured_source
    {

    public:
        enum EmitterType
        {
            directional,
            diffuse
        }; // diffuse is still not implemented
        area_textured_source(const vector_type<Float> &origin, const vector_type<Float> &dir, const Float &halfThetaLimit, const std::string &filename,
                        const tvec::Vec2f &plane, const Float &Li, const vector_type<Float> &lens_origin, const Float &lens_aperture, const Float &lens_focalLength, const bool &lens_active, const EmitterType &emittertype = EmitterType::directional)
            : m_origin(origin),
            m_dir(dir),
            m_half_theta_limit(halfThetaLimit),
            m_emittertype(emittertype),
            m_plane(plane),
            m_Li(Li),
            m_lens(lens_origin, lens_aperture, lens_focalLength, lens_active)
        {
            this->m_texture.readFile(filename);
            int _length = this->m_texture.getXRes() * this->m_texture.getYRes();
            this->m_pixelsize.x = this->m_plane.x / this->m_texture.getXRes();
            this->m_pixelsize.y = this->m_plane.y / this->m_texture.getYRes();

            this->m_ct = std::cos(this->m_half_theta_limit);

            this->m_texture_sampler.reserve(_length);
            for (int i = 0; i < _length; i++)
            {
                this->m_texture_sampler.append(this->m_texture.getPixel(i));
            }
            this->m_texture_sampler.normalize();
        }

        bool sample_ray(vector_type<Float> &pos, vector_type<Float> &dir, smp::Sampler &sampler, Float &totalDistance) const;

        inline const vector_type<Float> &get_origin() const
        {
            return this->m_origin;
        }

        inline const vector_type<Float> &get_dir() const
        {
            return this->m_dir;
        }

        inline const tvec::Vec2f &get_plane() const
        {
            return this->m_plane;
        }

        inline const Float getHalfThetaLimit() const
        {
            return this->m_half_theta_limit;
        }

        inline const EmitterType &getEmitterType() const
        {
            return this->m_emittertype;
        }

        inline const image::Texture &getTexture() const
        {
            return this->m_texture;
        }

        inline const Float get_li() const
        {
            return this->m_Li;
        }

        inline const Lens<vector_type> &getLens() const
        {
            return this->m_lens;
        }

        inline const std::vector<Float> &textureSamplerCDF() const
        {
            return this->m_texture_sampler.getCDF();
        }

        inline const bool propagateTillMedium(vector_type<Float> &pos, vector_type<Float> &dir, Float &totalDistance) const
        {
            // propagate till lens
            if (!this->m_lens.propagateTillLens(pos, dir, totalDistance))
                return false;
            return true;
        }

        virtual ~area_textured_source() {}

    protected:
        vector_type<Float> m_origin;
        vector_type<Float> m_dir;
        Float m_half_theta_limit;
        Float m_ct;
        image::Texture m_texture;
        DiscreteDistribution m_texture_sampler;
        tvec::Vec2f m_pixelsize;
        tvec::Vec2f m_plane;
        Float m_Li;
        EmitterType m_emittertype;
        Lens<vector_type> m_lens;
    };
}