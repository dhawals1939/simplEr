#pragma once
#include <string>
#include <sstream>

#include <rif.h>
#include <json_helper.h> // AnyMap/get_num/get_exact

class rif_interpolated final : public rif
{
public:
    Float f_u = 848 * 1e3;
    Float speed_u = 1500;
    Float n_o = 1.3333;
    std::string rifgrid_file = "us";

    rif_interpolated() = default;

    explicit rif_interpolated(const AnyMap &p)
    {
        this->f_u = get_num<Float>(p, "f_u");
        this->speed_u = get_num<Float>(p, "speed_u");
        this->n_o = get_num<Float>(p, "n_o");
        this->rifgrid_file = get_exact<std::string>(p, "rifgrid_file");
    }

    rif_interpolated(Float f_u_, Float speed_u_, Float n_o_, std::string rifgrid_file_)
        : f_u(f_u_), speed_u(speed_u_), n_o(n_o_), rifgrid_file(std::move(rifgrid_file_)) {}

    std::string to_string() const override
    {
        std::ostringstream oss;
        oss << "rif_interpolated:\n";
        oss << "  f_u = " << this->f_u << "\n";
        oss << "  speed_u = " << this->speed_u << "\n";
        oss << "  n_o = " << this->n_o << "\n";
        oss << "  rifgrid_file = " << this->rifgrid_file << "\n";
        return oss.str();
    }
};
