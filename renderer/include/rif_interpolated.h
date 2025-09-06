#pragma once
#include <string>
#include <sstream>

#include <rif.h>
#include <json_helper.h> // AnyMap/get_num/get_exact

class rif_interpolated final : public rif
{
public:
    Float m_f_u = 848 * 1e3;
    Float m_speed_u = 1500;
    Float m_n_o = 1.3333;
    std::string m_rifgrid_file = "us";

    rif_interpolated() = default;

    explicit rif_interpolated(const AnyMap &p)
    {
        this->m_f_u = get_num<Float>(p, "f_u");
        this->m_speed_u = get_num<Float>(p, "speed_u");
        this->m_n_o = get_num<Float>(p, "n_o");
        this->m_rifgrid_file = get_exact<std::string>(p, "rifgrid_file");
    }

    rif_interpolated(Float f_u_, Float speed_u_, Float n_o_, std::string rifgrid_file_)
        : m_f_u(f_u_), m_speed_u(speed_u_), m_n_o(n_o_), m_rifgrid_file(std::move(rifgrid_file_)) {}

    friend std::ostream& operator<<(std::ostream& os, const rif_interpolated& obj)
    {
        os << "rif_interpolated:\n"
           << "  f_u = " << obj.m_f_u << "\n"
           << "  speed_u = " << obj.m_speed_u << "\n"
           << "  n_o = " << obj.m_n_o << "\n"
           << "  rifgrid_file = " << obj.m_rifgrid_file << "\n";
        return os;
    }
};
