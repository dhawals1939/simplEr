#pragma once

namespace scn
{
    template <template <typename> class vector_type>
    class Block
    {

    public:
        Block(const vector_type<Float> &blockL, const vector_type<Float> &blockR)
            : m_blockL(blockL),
            m_blockR(blockR) {}

        /*
        * TODO: Maybe replace these with comparisons to FPCONST(0.0)?
        */
        inline bool inside(const vector_type<Float> &p) const
        {
            bool result = true;
            for (int iter = 0; iter < p.dim; ++iter)
            {
                result = result && (p[iter] - this->m_blockL[iter] > -M_EPSILON) && (this->m_blockR[iter] - p[iter] > -M_EPSILON);
                /*
                * TODO: Maybe remove this check, it may be slowing performance down
                * due to the branching.
                */
                if (!result)
                {
                    break;
                }
            }
            return result;
        }

        bool intersect(const vector_type<Float> &p, const vector_type<Float> &d,
                    Float &disx, Float &disy) const;

        inline const vector_type<Float> &getBlockL() const
        {
            return this->m_blockL;
        }

        inline const vector_type<Float> &getBlockR() const
        {
            return this->m_blockR;
        }

        virtual ~Block() {}

    protected:
        vector_type<Float> m_blockL;
        vector_type<Float> m_blockR;
    };
};