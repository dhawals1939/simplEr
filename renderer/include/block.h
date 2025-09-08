#include <constants.h>
#include <tvector.h>
#include <cstddef>

#pragma once
namespace scn
{
    template <template <typename> class vector_type>
    class block
    {
    public:
        block(const vector_type<Float>& blockL, const vector_type<Float>& blockR);

        bool inside(const vector_type<Float>& p) const;

        bool intersect(const vector_type<Float>& p, const vector_type<Float>& d,
                       Float& disx, Float& disy) const;

        const vector_type<Float>& get_block_l() const;
        const vector_type<Float>& get_block_r() const;

        ~block();

        block& operator<<(const vector_type<Float>& p);

    protected:
        vector_type<Float> m_blockL;
        vector_type<Float> m_blockR;
    };
} // namespace scn