#include <fmt/core.h>
#include <boost/algorithm/string.hpp>
#include <boost/core/demangle.hpp>
#include <boost/static_assert.hpp>
#include <boost/random.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/math/constants/constants.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <iostream>
#include <openblas/cblas.h>

template <typename T, int Rows, int Cols>
struct fmt::formatter<Eigen::Matrix<T, Rows, Cols>>
{
    constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const Eigen::Matrix<T, Rows, Cols> &mat, FormatContext &ctx) const
    { // Marked as const
        auto out = ctx.out();
        for (int i = 0; i < mat.rows(); ++i)
        {
            for (int j = 0; j < mat.cols(); ++j)
            {
                out = fmt::format_to(out, "{} ", mat(i, j));
            }
            out = fmt::format_to(out, "\n");
        }
        return out;
    }
};

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
    fmt::print("Eigen3 Matrix:\n{}\n", mat);

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

    fmt::print("Initial x: {}\n", initial_x);
    fmt::print("Final x: {}\n", x);

    // Test CBLAS
    int m = 2, n = 2, k = 2;
    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4] = {0, 0, 0, 0};

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C, n);

    fmt::print("CBLAS Matrix Multiplication Result:\n");
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            fmt::print("{} ", C[i * n + j]);
        }
        fmt::print("\n");
    }

    // Example usage of boost::core::demangle
    fmt::print("Demangled name: {}\n", boost::core::demangle(typeid(int).name()));

    // Example usage of boost::static_assert
    BOOST_STATIC_ASSERT(sizeof(int) == 4);

    // Example usage of boost::random
    boost::random::mt19937 rng;
    boost::random::uniform_int_distribution<> dist(1, 100);
    fmt::print("Random number: {}\n", dist(rng));

    // Example usage of boost::math::constants
    fmt::print("Pi: {}\n", boost::math::constants::pi<double>());

    return 0;
}
