#pragma once

namespace scn {
template <template <typename> class vector_type>
class Lens
{

public:
    Lens(const vector_type<Float> &origin, const Float &aperture,
         const Float &focalLength, const bool &active)
        : m_origin(origin),
          m_squareApertureRadius(aperture * aperture),
          m_focalLength(focalLength),
          m_active(active)
    {
    }
    inline const bool propagateTillLens(vector_type<Float> &pos, vector_type<Float> &dir, Float &totalDistance) const
    {
        Float dist = -(pos[0] - this->m_origin[0]) / dir[0]; // FIXME: Assumes that the direction of propagation is in -x direction.
        pos += dist * dir;
        totalDistance += dist;
        if (this->m_active)
            return this->deflect(pos, dir, totalDistance);
        else
            return true;
    }

    inline const bool isActive() const
    {
        return this->m_active;
    }

    inline const vector_type<Float> &get_origin() const
    {
        return this->m_origin;
    }

    inline const Float getSquareApertureRadius() const
    {
        return this->m_squareApertureRadius;
    }

    inline const Float getFocalLength() const
    {
        return this->m_focalLength;
    }
    inline const bool deflect(const vector_type<Float> &pos, vector_type<Float> &dir, Float &totalDistance) const
    {
        /* Deflection computation:
         * Point going through the center of lens and parallel to dir is [pos.x, 0, 0]. Ray from this point goes straight
         * This ray meets focal plane at (pos.x - d[0] * f/d[0], -d[1] * f/d[0], -d[2] * f/d[0]) (assuming -x direction of propagation of light)
         * Original ray deflects to pass through this point
         * The negative distance (HACK) travelled by this ray at the lens is -f/d[0] - norm(focalpoint_Pos - original_Pos)
         */
        Float squareDistFromLensOrigin = 0.0f;
        for (int i = 1; i < pos.dim; i++)
            squareDistFromLensOrigin += pos[i] * pos[i];
        if (squareDistFromLensOrigin > this->m_squareApertureRadius)
            return false;

        Assert(pos.x == this->m_origin.x);
        Float invd = -1 / dir[0];
        dir[0] = -this->m_focalLength;
        dir[1] = dir[1] * invd * this->m_focalLength - pos[1];
        dir[2] = dir[2] * invd * this->m_focalLength - pos[2];
        totalDistance += this->m_focalLength * invd - dir.length();
        dir.normalize();
        return true; // should return additional path length added by the lens.
    }

protected:
    vector_type<Float> m_origin;
    Float m_squareApertureRadius; // radius of the aperture
    Float m_focalLength;
    bool m_active; // Is the lens present or absent
};
}