#include <block.h>

namespace scn
{
    template <template <typename> class vector_type>
    block<vector_type>::block(const vector_type<Float>& blockL,
                              const vector_type<Float>& blockR)
        : m_blockL(blockL), m_blockR(blockR)
    {}

    template <template <typename> class vector_type>
    bool block<vector_type>::inside(const vector_type<Float>& p) const
    {
        bool result = true;
        for (int iter = 0; iter < p.dim; ++iter)
        {
            result = result
                && (p[iter] - this->m_blockL[iter] > -M_EPSILON)
                && (this->m_blockR[iter] - p[iter] > -M_EPSILON);
            if (!result) break;
        }
        return result;
    }

    template <template <typename> class vector_type>
    bool block<vector_type>::intersect(const vector_type<Float>& /*p*/,
                                       const vector_type<Float>& /*d*/,
                                       Float& /*disx*/, Float& /*disy*/) const
    {
        // TODO: implement your real intersection; returning false keeps it linkable.
        return false;
    }

    template <template <typename> class vector_type>
    const vector_type<Float>& block<vector_type>::get_block_l() const
    {
        return this->m_blockL;
    }

    template <template <typename> class vector_type>
    const vector_type<Float>& block<vector_type>::get_block_r() const
    {
        return this->m_blockR;
    }

    template <template <typename> class vector_type>
    block<vector_type>::~block() = default;


    template <template <typename> class vector_type>
    std::ostream& operator<<(std::ostream& os, const block<vector_type>& blk)
    {
        os << "block(L: " << blk.get_block_l() << ", us_wave_radius: " << blk.get_block_r() << ")";
        return os;
    }

    // ---------- Explicit template instantiations ----------
    // Add a line for every vector_type you actually use in the program.
    template class block<tvec::TVector3>;
    template class block<tvec::TVector2>;
} // namespace scn