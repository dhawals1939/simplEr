#pragma once
#include <string>
#include <constants.h>

class rif
{
public:
    virtual ~rif() = default;
    friend std::ostream& operator<<(std::ostream& os, const rif& obj)
    {
        return os << "rif instance";
    }
};
