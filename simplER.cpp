#include <fmt/core.h>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <iostream>

int main()
{
    // Test fmt
    fmt::print("Hello World!\n");

    // Test Boost
    std::string s = "Boost Library";
    boost::to_upper(s);
    fmt::print("Boost: {}\n", s);

    // Test Eigen3
    Eigen::MatrixXd mat(2, 2);
    mat(0, 0) = 3;
    mat(1, 0) = 2.5;
    mat(0, 1) = -1;
    mat(1, 1) = mat(1, 0) + mat(0, 1);
    std::cout << "Eigen3 Matrix:\n"
              << mat << std::endl;

    return 0;
}