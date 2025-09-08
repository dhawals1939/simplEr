#include <constants.h>
#include <sampler.h>
#include <tvector.h>
#include <image.h>
#include <lens.h>
#include <pmf.h>
#include <cmath>
#include <vector>
#include <cstddef>

#pragma once
namespace scn
{
    template <template <typename> class vector_type>
    class area_textured_source
    {
    public:
        enum EmitterType { directional, diffuse }; // diffuse not implemented

        area_textured_source(const vector_type<Float>& origin,
                             const vector_type<Float>& dir,
                             const Float& half_theta_limit,
                             const std::string& filename,
                             const tvec::Vec2f& plane,
                             const Float& Li,
                             const vector_type<Float>& lens_origin,
                             const Float& lens_aperture,
                             const Float& lens_focal_length,
                             const bool&  lens_active,
                             const EmitterType& emittertype = EmitterType::directional);

        bool sample_ray(vector_type<Float>& pos,
                        vector_type<Float>& dir,
                        smp::Sampler& sampler,
                        Float& total_distance) const;

        const vector_type<Float>& get_origin() const;
        const vector_type<Float>& get_dir() const;
        const tvec::Vec2f&       get_plane() const;
        const Float              get_half_theta_limit() const;
        const EmitterType&       get_emitter_type() const;
        const image::Texture&    get_texture() const;
        const Float              get_Li() const;
        const Lens<vector_type>& get_lens() const;
        const std::vector<Float>& texture_sampler_cdf() const;

        const bool propagate_till_medium(vector_type<Float>& pos,
                                         vector_type<Float>& dir,
                                         Float& total_distance) const;

        virtual ~area_textured_source();

        template <template <typename> class V>
        friend std::ostream& operator<<(std::ostream& os, const area_textured_source<V>& src);

    protected:
        vector_type<Float>   m_origin;
        vector_type<Float>   m_dir;
        Float                m_half_theta_limit;
        Float                m_ct;
        image::Texture       m_texture;
        DiscreteDistribution m_texture_sampler;
        tvec::Vec2f          m_pixelsize;
        tvec::Vec2f          m_plane;
        Float                m_Li;
        EmitterType          m_emittertype;
        Lens<vector_type>    m_lens;
    };
} // namespace scn
