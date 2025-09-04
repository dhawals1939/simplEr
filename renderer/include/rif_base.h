#pragma once
#include <string>
#include <constants.h>

class rif
{
public:
    virtual ~rif() = default;
    virtual std::string to_string() const = 0; // pretty-printer
};
