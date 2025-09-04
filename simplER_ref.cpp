// main.cpp
#include <renderer.h>
#include <util.h>
#include <sampler.h>
#include <vmf.h>
#include <regex>
#include <sstream>
#include <image.h>
#include <vector>
#include <json_helper.h>
#include <fmt/core.h>

#include <fstream>
#include <ctime>
#include <memory>

#include <config.h>     // GlobalConfig + load_global_config
#include <rif_config.h> // RifConfig + print()

/* ADITHYA: Known issues/inconsistencies to be fixed
 * 1. MaxPathLength and PathLengthRange's maxPathLength are inconsistent !!
 * 2. Timing information for the ER is not accurate
 * 3. IOR and n_o are inconsistent
 * 4. Differential rendering part is completely broken
 * 5. 2D rendering is broken
 */

int main(int argc, char **argv)
{
    std::string config_file = (argc > 1) ? argv[1] : "config.json";
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "Using config file: {}\n", config_file);

    std::ifstream file(config_file);
    if (!file.is_open())
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), "Failed to open {}\n", config_file);
        return EXIT_FAILURE;
    }

    nlohmann::json j;
    file >> j;

    AnyMap cfg;
    convertJsonToMap(j, cfg);
    if (cfg.empty())
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), "Failed to parse {}\n", config_file);
        return EXIT_FAILURE;
    }

    GlobalConfig C;
    try
    {
        C = load_global_config(cfg);
    }
    catch (const std::exception &e)
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red),
                   "Error occurred while starting simplER renderer: {}\n", e.what());
        if (C.stricts)
            return EXIT_FAILURE;
    }

    // Print selected inputs (mirrors your previous prints; add/remove as you like)
    if (C.print_inputs)
    {
        fmt::print(fmt::fg(fmt::color::green), "num_photons = {}\n", C.num_photons);
        fmt::print(fmt::fg(fmt::color::green), "out_file_prefix = {}\n", C.out_file_prefix);
        fmt::print(fmt::fg(fmt::color::green), "sigma_t = {}\n", C.sigma_t);
        fmt::print(fmt::fg(fmt::color::green), "albedo = {}\n", C.albedo);
        fmt::print(fmt::fg(fmt::color::green), "g_val = {}\n", C.g_val);
        print(C.rif);

        fmt::print(fmt::fg(fmt::color::green), "emitter_gap = {}\n", C.emitter_gap);
        fmt::print(fmt::fg(fmt::color::green), "sensor_gap = {}\n", C.sensor_gap);
        fmt::print(fmt::fg(fmt::color::green), "projector_texture = {}\n", C.projector_texture);
        fmt::print(fmt::fg(fmt::color::green), "use_direct = {}\n", C.use_direct);
        fmt::print(fmt::fg(fmt::color::green), "use_angular_sampling = {}\n", C.use_angular_sampling);
        fmt::print(fmt::fg(fmt::color::green), "use_bounce_decomposition = {}\n", C.use_bounce_decomposition);
        fmt::print(fmt::fg(fmt::color::green), "max_depth = {}\n", C.max_depth);
        fmt::print(fmt::fg(fmt::color::green), "max_pathlength = {}\n", C.max_pathlength);
        fmt::print(fmt::fg(fmt::color::green), "Total medium length = {}\n", C.medium_r[0] - C.medium_l[0]);
        fmt::print(fmt::fg(fmt::color::green), "pathLengthMin = {}\n", C.path_length_min);
        fmt::print(fmt::fg(fmt::color::green), "pathLengthMax = {}\n", C.path_length_max);
        fmt::print(fmt::fg(fmt::color::green), "pathLengthBins = {}\n", C.path_length_bins);
        fmt::print(fmt::fg(fmt::color::green), "spatialX = {}\n", C.spatial_x);
        fmt::print(fmt::fg(fmt::color::green), "spatialY = {}\n", C.spatial_y);
        fmt::print(fmt::fg(fmt::color::green), "halfThetaLimit = {}\n", C.half_theta_limit);
        fmt::print(fmt::fg(fmt::color::green), "emitter_size = {}\n", C.emitter_size);
        fmt::print(fmt::fg(fmt::color::green), "sensor_size = {}\n", C.sensor_size);
        fmt::print(fmt::fg(fmt::color::green), "distribution = {}\n", C.distribution);
        fmt::print(fmt::fg(fmt::color::green), "gOrKappa = {}\n", C.g_or_kappa);
        fmt::print(fmt::fg(fmt::color::green), "emitter_distance = {}\n", C.emitter_distance);
        fmt::print(fmt::fg(fmt::color::green), "emitter_lens_aperture = {}\n", C.emitter_lens_aperture);
        fmt::print(fmt::fg(fmt::color::green), "emitter_lens_focal_length = {}\n", C.emitter_lens_focal_length);
        fmt::print(fmt::fg(fmt::color::green), "emitter_lens_active = {}\n", C.emitter_lens_active);
        fmt::print(fmt::fg(fmt::color::green), "sensor_distance = {}\n", C.sensor_distance);
        fmt::print(fmt::fg(fmt::color::green), "sensor_lens_aperture = {}\n", C.sensor_lens_aperture);
        fmt::print(fmt::fg(fmt::color::green), "sensor_lens_focal_length = {}\n", C.sensor_lens_focal_length);
        fmt::print(fmt::fg(fmt::color::green), "sensor_lens_active = {}\n", C.sensor_lens_active);
        fmt::print(fmt::fg(fmt::color::green), "printInputs = {}\n", C.print_inputs);
    }

    // -----------------------
    // Sanity checks (same logic as before)
    // -----------------------
    const Float medium_len_x = C.medium_r[0] - C.medium_l[0];

    if (C.emitter_distance < 0)
    {
        std::cout << "emitter_distance = " << C.emitter_distance << " should be strictly non-zero" << std::endl;
        return EXIT_FAILURE;
    }
    if (C.sensor_distance < 0)
    {
        std::cout << "sensor_distance = " << C.sensor_distance << " should be strictly non-zero" << std::endl;
        return EXIT_FAILURE;
    }
    if (C.emitter_lens_active && C.emitter_distance < 1e-4)
    {
        std::cout << "lens_active and emitter_distance = " << C.emitter_distance
                  << ". emitter_distance should be strictly positive (>1e-4) " << std::endl;
        return EXIT_FAILURE;
    }
    if (C.sensor_lens_active && C.sensor_distance < 1e-4)
    {
        std::cout << "lens_active and sensor_distance = " << C.sensor_distance
                  << ". sensor_distance should be strictly positive (>1e-4) " << std::endl;
        return EXIT_FAILURE;
    }
    if (C.emitter_gap < 0 || C.emitter_gap > medium_len_x)
    {
        std::cout << "invalid gap between the emitter and the US:" << C.emitter_gap << std::endl;
        return EXIT_FAILURE;
    }
    if (C.sensor_gap < 0 || C.sensor_gap > medium_len_x)
    {
        std::cout << "invalid gap between the sensor and the US:" << C.sensor_gap << std::endl;
        return EXIT_FAILURE;
    }
    if ((C.sensor_gap + C.emitter_gap) > medium_len_x)
    {
        std::cout << "sum of sensor and emitter gaps is more than the medium size; sum of gaps is "
                  << (C.sensor_gap + C.emitter_gap) << std::endl;
        return EXIT_FAILURE;
    }
    if (C.rif.kind == RifKind::Basic)
    {
        const auto &rb = std::get<RifBasic>(C.rif.data);
        if (rb.phi_max < rb.phi_min)
        {
            std::cout << "phi_max must be greater than or equal to phi_min" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // -----------------------
    // Build scene + render
    // -----------------------
    pfunc::HenyeyGreenstein *phase = new pfunc::HenyeyGreenstein(C.g_val);

    tvec::Vec3f emitter_lens_origin(C.medium_r[0], FPCONST(0.0), FPCONST(0.0));
    Float EgapEndLocX = emitter_lens_origin.x - C.emitter_gap;
    tvec::Vec3f sensor_lens_origin(C.medium_l[0], FPCONST(0.0), FPCONST(0.0));
    Float SgapBeginLocX = sensor_lens_origin.x + C.sensor_gap; // ADI: VERIFY ME

    // Source (emitter)
    const tvec::Vec3f lightOrigin(C.medium_r[0] + C.emitter_distance, FPCONST(0.0), FPCONST(0.0));
    const Float lightAngle = FPCONST(M_PI);
    const tvec::Vec3f lightDir(std::cos(lightAngle), std::sin(lightAngle), FPCONST(0.0));
    const tvec::Vec2f lightPlane(C.emitter_size, C.emitter_size);
    const Float Li = FPCONST(75000.0);

    // Camera (sensor)
    const tvec::Vec3f viewOrigin(C.medium_l[0] - C.sensor_distance, FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f viewDir(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f viewX(FPCONST(0.0), -FPCONST(1.0), FPCONST(0.0));
    const tvec::Vec2f viewPlane(C.sensor_size, C.sensor_size);
    const tvec::Vec2f pathlengthRange(C.path_length_min, C.path_length_max);
    const tvec::Vec3i viewReso(C.spatial_x, C.spatial_y, C.path_length_bins);

    // Rendering params
    const tvec::Vec3f axis_uz(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f axis_ux(FPCONST(0.0), FPCONST(0.0), FPCONST(1.0));
    const tvec::Vec3f p_u(FPCONST(0.0), FPCONST(0.0), FPCONST(0.0));

    const med::Medium medium(C.sigma_t, C.albedo, phase);

    std::unique_ptr<scn::Scene<tvec::TVector3>> scenePtr;

    // Dispatch to the proper Scene<> constructor based on the active RIF type
    std::visit([&](const auto &rifData)
               {
        using T = std::decay_t<decltype(rifData)>;

        if constexpr (std::is_same_v<T, RifFus>) {
            // Ensure build supports FUS
#if USE_RIF_SOURCES
            scenePtr = std::make_unique<scn::Scene<tvec::TVector3>>(
                C.ior, C.medium_l, C.medium_r,
                lightOrigin, lightDir, C.half_theta_limit, C.projector_texture, lightPlane, Li,
                viewOrigin, viewDir, viewX, viewPlane, pathlengthRange, C.use_bounce_decomposition,
                C.distribution, C.g_or_kappa,
                emitter_lens_origin, C.emitter_lens_aperture, C.emitter_lens_focal_length, C.emitter_lens_active,
                sensor_lens_origin, C.sensor_lens_aperture, C.sensor_lens_focal_length, C.sensor_lens_active,
                // FUS-specific arguments:
                rifData.f_u, rifData.speed_u, rifData.n_o, rifData.n_scaling, rifData.n_coeff, rifData.radius,
                rifData.center1, rifData.center2, rifData.active1, rifData.active2, rifData.phase1, rifData.phase2,
                rifData.theta_min, rifData.theta_max, rifData.theta_sources, rifData.trans_z_min, rifData.trans_z_max,
                rifData.trans_z_sources,
                axis_uz, axis_ux, p_u, C.er_stepsize, C.direct_to_l, C.rr_weight, C.precision, EgapEndLocX, SgapBeginLocX, C.use_initialization_hack
            );
#else
            fmt::print(fmt::fg(fmt::color::red),
                       "Requested RIF type 'fus' but the build is not compiled with USE_RIF_SOURCES.\n");
            std::exit(EXIT_FAILURE);
#endif
        } else if constexpr (std::is_same_v<T, RifSpline>) {
            // Ensure build supports SPLINE
#if USE_RIF_INTERPOLATED
            scenePtr = std::make_unique<scn::Scene<tvec::TVector3>>(
                C.ior, C.medium_l, C.medium_r,
                lightOrigin, lightDir, C.half_theta_limit, C.projector_texture, lightPlane, Li,
                viewOrigin, viewDir, viewX, viewPlane, pathlengthRange, C.use_bounce_decomposition,
                C.distribution, C.g_or_kappa,
                emitter_lens_origin, C.emitter_lens_aperture, C.emitter_lens_focal_length, C.emitter_lens_active,
                sensor_lens_origin, C.sensor_lens_aperture, C.sensor_lens_focal_length, C.sensor_lens_active,
                // BASIC-or-FUS section in your original signature is replaced by SPLINE builds at compile time.
                // Keep the order exactly like your previous #if USE_RIF_INTERPOLATED block:
                axis_uz, axis_ux, p_u, C.er_stepsize, C.direct_to_l, C.rr_weight, C.precision, EgapEndLocX, SgapBeginLocX, C.use_initialization_hack,
                rifData.rifgrid_file
            );
#else
            fmt::print(fmt::fg(fmt::color::red),
                       "Requested RIF type 'spline' but the build is not compiled with USE_RIF_INTERPOLATED.\n");
            std::exit(EXIT_FAILURE);
#endif
        } else { // RifBasic
            // Basic path (no special compile flag required)
            scenePtr = std::make_unique<scn::Scene<tvec::TVector3>>(
                C.ior, C.medium_l, C.medium_r,
                lightOrigin, lightDir, C.half_theta_limit, C.projector_texture, lightPlane, Li,
                viewOrigin, viewDir, viewX, viewPlane, pathlengthRange, C.use_bounce_decomposition,
                C.distribution, C.g_or_kappa,
                emitter_lens_origin, C.emitter_lens_aperture, C.emitter_lens_focal_length, C.emitter_lens_active,
                sensor_lens_origin, C.sensor_lens_aperture, C.sensor_lens_focal_length, C.sensor_lens_active,
                // BASIC-specific arguments:
                rifData.f_u, rifData.speed_u, rifData.n_o, rifData.n_max, rifData.n_clip, rifData.phi_min, rifData.phi_max, rifData.mode,
                axis_uz, axis_ux, p_u, C.er_stepsize, C.direct_to_l, C.rr_weight, C.precision, EgapEndLocX, SgapBeginLocX, C.use_initialization_hack
            );
        } }, C.rif.data);

    photon::Renderer<tvec::TVector3> renderer(C.max_depth, C.max_pathlength, C.use_direct, C.use_angular_sampling, C.threads);

    image::SmallImage img(viewReso.x, viewReso.y, viewReso.z);
    renderer.renderImage(img, medium, *scenePtr, C.num_photons);

    img.writePFM3D(C.out_file_prefix + std::string(".pfm3d"));

    delete phase;
    return 0;
}
