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
#include <mex.h>

// Formatter for Eigen matrices
template <typename T, int Rows, int Cols>
struct fmt::formatter<Eigen::Matrix<T, Rows, Cols>>
{
    constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const Eigen::Matrix<T, Rows, Cols> &mat, FormatContext &ctx) const
    {
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

// Simple cost function for Ceres
struct CostFunctor
{
    template <typename T>
    bool operator()(const T *const x, T *residual) const
    {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};

void TestDenseLinearAlgebraLibraries();

// A simple cost function for testing
struct QuadraticCostFunctor
{
    template <typename T>
    bool operator()(const T *const x, T *residual) const
    {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};

void TestDenseLinearAlgebraLibraries()
{
    fmt::print("\nTesting Dense Linear Algebra Libraries in Ceres...\n");

    // Define the problem
    double x = 0.5; // Initial guess
    ceres::Problem problem;

    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<QuadraticCostFunctor, 1, 1>(new QuadraticCostFunctor);
    problem.AddResidualBlock(cost_function, nullptr, &x);

    // Test different dense linear algebra libraries
    const std::vector<ceres::DenseLinearAlgebraLibraryType> libraries = {
        ceres::EIGEN,
        ceres::CUDA, ceres::LAPACK};

    for (auto library : libraries)
    {
        // Solver options
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.linear_solver_type = ceres::DENSE_QR;
        options.dense_linear_algebra_library_type = library;

        // Solve the problem
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Print results
        const char *library_name = (library == ceres::EIGEN) ? "EIGEN" : (library == ceres::LAPACK) ? "LAPACK"
                                                                                                    : "CUDA";
        fmt::print("\nResults with {}:\n", library_name);
        fmt::print("Initial x: {}\n", 0.5);
        fmt::print("Final x: {}\n", x);
        fmt::print("{}\n", summary.BriefReport());
    }
    fmt::print("\n");
}

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

    // Test Ceres optimization
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

    TestDenseLinearAlgebraLibraries();

    return 0;
}
