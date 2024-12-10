#include <fmt/core.h>
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <iostream>
#include <openblas/cblas.h>

// A simple cost function for Ceres
struct CostFunctor
{
    template <typename T>
    bool operator()(const T *const x, T *residual) const
    {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};

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

    // Test Ceres
    double initial_x = 5.0;
    double x = initial_x;

    ceres::Problem problem;
    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, nullptr, &x);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Initial x: " << initial_x << "\n";
    std::cout << "Final x: " << x << "\n";

    // Test CBLAS
    int m = 2, n = 2, k = 2;
    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4] = {0, 0, 0, 0};

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);

    std::cout << "CBLAS Matrix Multiplication Result:\n";
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}