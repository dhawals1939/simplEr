#pragma once
#include <tvector.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <any>
#include <map>
#include <string>
#include <vector>
#include <constants.h>
#include <json_helper.h>
#include <rif.h>

// adoc geometry parameters
typedef struct adoc_geometry_parameters
{
    Float half_theta_limit = FPCONST(12.8e-3);
    Float emitter_size = FPCONST(0.002);   // emitter side length (square)
    Float sensor_size = FPCONST(0.002);    // sensor side length (square)
    Float emitter_distance = FPCONST(0.0); // distance from origin
    Float sensor_distance = FPCONST(0.0);  // distance from origin
};

// Film parameters
typedef struct film_parameters{
    Float path_length_min = FPCONST(0.0);  // minimum path length bin
    Float path_length_max = FPCONST(64.0); // maximum path length bin
    int path_length_bins = 128;            // number of path length bins
    int spatial_x = 128;                   // film resolution (x)
    int spatial_y = 128;                   // film resolution (y)
};


// Scattering parameters
typedef struct scattering_parameters
{
    Float sigma_t = FPCONST(0.0);
    Float albedo = FPCONST(1.0);
    Float g_val = FPCONST(0.0);
};

// Rendering parameters
typedef struct rendering_parameters
{
    int64 num_photons = 500L;
    int max_depth = -1;
    Float max_pathlength = -1;
    bool use_direct = false;
    bool use_angular_sampling = true;
};


// Scene parameters
typedef struct scene_parameters
{
    tvec::Vec3f medium_l = tvec::Vec3f(-FPCONST(.015), -FPCONST(5.0), -FPCONST(5.0));
    tvec::Vec3f medium_r = tvec::Vec3f(FPCONST(.015), FPCONST(5.0), FPCONST(5.0));
};


// Importance sampling parameters
typedef struct importance_sampling_parameters {
    std::string distribution = "vmf"; // options: vmf, hg, uniform, none
    Float g_or_kappa = 4;
};

// Lens parameters
typedef struct lens_parameters {
    Float emitter_lens_aperture = .015;
    Float emitter_lens_focal_length = .015;
    bool emitter_lens_active = false;
    Float sensor_lens_aperture = .015;
    Float sensor_lens_focal_length = .015;
    bool sensor_lens_active = false;
};


typedef struct settings
{
    // Output and system parameters
    std::string rendering_type = "analytic_rif";
    std::string output_file_name = "";
    int threads = -1; // default number of threads

    // adoc geometry parameters
    adoc_geometry_parameters adoc_geometry_params;
    // scattering parameters
    scattering_parameters scattering_params;
    // rendering parameters
    rendering_parameters rendering_params;
    // scene parameters
    scene_parameters scene_params;
    // film parameters
    film_parameters film_params;
    // importance sampling parameters
    importance_sampling_parameters importance_sampling_params;
    // lens parameters
    lens_parameters lens_params;
    // rif parameters
    std::unique_ptr<rif> rif_params; // parent class handle

    // Extended rendering parameters
    Float emitter_gap = .0; // distance before US activation (from emitter)
    Float sensor_gap = .0;  // distance before US activation (towards sensor)
    Float er_stepsize = 1e-3;
    int precision = 8; // decimal precision for ER stepping
    Float direct_to_l = 1e-5;
    bool use_initialization_hack = true; // force direct connections to line between endpoints
    Float rr_weight = 1e-2;              // Russian roulette survival probability

    bool use_bounce_decomposition = true; // true = bounce decomposition, false = transient
    bool print_inputs = true;
    std::string projector_texture = "/home/dhawals1939/repos/simplER/renderer/images/White.pfm";
};

void parse_config(const AnyMap &config, struct settings &settings, bool &stricts)
{
    try
    {
        // ---- flat / top-level ----
        stricts = get_exact<bool>(config, "stricts");
        settings.rendering_type = get_exact<std::string>(config, "rendering_type");
        settings.threads = get_num<int>(config, "threads");
        settings.precision = get_num<int>(config, "precision");

        settings.emitter_gap = get_num<Float>(config, "emitter_gap");
        settings.sensor_gap = get_num<Float>(config, "sensor_gap");
        settings.er_stepsize = get_num<Float>(config, "er_stepsize");
        settings.direct_to_l = get_num<Float>(config, "direct_to_l");
        settings.use_initialization_hack = get_exact<bool>(config, "use_initialization_hack");
        settings.rr_weight = get_num<Float>(config, "rr_weight");
        settings.use_bounce_decomposition = get_exact<bool>(config, "use_bounce_decomposition");
        settings.print_inputs = get_exact<bool>(config, "print_inputs");
        settings.projector_texture = get_exact<std::string>(config, "projector_texture");

        // ---- validate + timestamp ----
        // create output_file_name
        {
            if (settings.rendering_type != "rif_analytical" &&
                settings.rendering_type != "rif_sources" &&
                settings.rendering_type != "rif_interpolated")
            {
                fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red),
                           "Invalid rendering_type: '{}'. Must be one of "
                           "'rif_analytical', 'rif_sources', or 'rif_interpolated'.\n",
                           settings.rendering_type);
                exit(EXIT_FAILURE);
            }
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            char datetime[32];
            std::strftime(datetime, sizeof(datetime), "%Y_%m_%d_%H_%M", &tm);
            settings.output_file_name = settings.rendering_type + "_";
            settings.output_file_name += datetime;
        }

        // ---- nested: film_parameters ----
        settings.film_params.path_length_min = get_num_path<Float>(config, "film_parameters.path_length.min");
        settings.film_params.path_length_max = get_num_path<Float>(config, "film_parameters.path_length.max");
        settings.film_params.path_length_bins = get_num_path<int>(config, "film_parameters.path_length.bins");
        settings.film_params.spatial_x = get_num_path<int>(config, "film_parameters.spatial.x");
        settings.film_params.spatial_y = get_num_path<int>(config, "film_parameters.spatial.y");

        // ---- nested: adoc_parameters ----
        settings.adoc_geometry_params.half_theta_limit = get_num_path<Float>(config, "adoc_parameters.half_theta_limit");
        settings.adoc_geometry_params.emitter_size = get_num_path<Float>(config, "adoc_parameters.emitter_size");
        settings.adoc_geometry_params.sensor_size = get_num_path<Float>(config, "adoc_parameters.sensor_size");
        settings.adoc_geometry_params.emitter_distance = get_num_path<Float>(config, "adoc_parameters.emitter_distance");
        settings.adoc_geometry_params.sensor_distance = get_num_path<Float>(config, "adoc_parameters.sensor_distance");

        // ---- nested: scattering_parameters ----
        settings.scattering_params.sigma_t = get_num_path<Float>(config, "scattering_parameters.sigma_t");
        settings.scattering_params.albedo = get_num_path<Float>(config, "scattering_parameters.albedo");
        settings.scattering_params.g_val = get_num_path<Float>(config, "scattering_parameters.g_val");

        // ---- nested: rendering_parameters ----
        settings.rendering_params.num_photons = get_num_path<int>(config, "rendering_parameters.num_photons");
        settings.rendering_params.use_direct = get_exact_path<bool>(config, "rendering_parameters.use_direct");
        settings.rendering_params.max_depth = get_num_path<int>(config, "rendering_parameters.max_depth");
        settings.rendering_params.max_pathlength = get_num_path<Float>(config, "rendering_parameters.max_pathlength");
        settings.rendering_params.use_angular_sampling = get_exact_path<bool>(config, "rendering_parameters.use_angular_sampling");

        // ---- nested: importance_sampling_parameters ----
        settings.importance_sampling_params.distribution = get_exact_path<std::string>(config, "importance_sampling_parameters.distribution");
        settings.importance_sampling_params.g_or_kappa = get_num_path<Float>(config, "importance_sampling_parameters.g_or_kappa");

        // ---- nested: scene_parameters ----
        settings.scene_params.medium_l[0] = get_num_path<Float>(config, "scene_parameters.medium_lx");
        settings.scene_params.medium_r[0] = get_num_path<Float>(config, "scene_parameters.medium_rx");

        // ---- nested: lens_parameters ----
        settings.lens_params.emitter_lens_aperture = get_num_path<Float>(config, "lens_parameters.emitter_lens_aperture");
        settings.lens_params.emitter_lens_focal_length = get_num_path<Float>(config, "lens_parameters.emitter_lens_focal_length");
        settings.lens_params.emitter_lens_active = get_exact_path<bool>(config, "lens_parameters.emitter_lens_active");
        settings.lens_params.sensor_lens_aperture = get_num_path<Float>(config, "lens_parameters.sensor_lens_aperture");
        settings.lens_params.sensor_lens_focal_length = get_num_path<Float>(config, "lens_parameters.sensor_lens_focal_length");
        settings.lens_params.sensor_lens_active = get_exact_path<bool>(config, "lens_parameters.sensor_lens_active");

        // ---- nested: rif_parameters ----

        const auto &rif_params = get_exact<AnyMap>(config, "rif_parameters");

        if (settings.rendering_type == "rif_sources")
        {
            settings.rif_params = std::make_unique<rif_sources>(rif_params); // ctor(AnyMap)
        }
        else if (settings.rendering_type == "rif_analytical")
        {
            settings.rif_params = std::make_unique<rif_analytical>(rif_params);
        }
        else if (settings.rendering_type == "rif_interpolated")
        {
            settings.rif_params = std::make_unique<rif_interpolated>(rif_params);
        }
    }
    catch (const std::exception &e)
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red),
                   "Error occurred while starting simplER renderer: {}\n", e.what());
        if (std::string(e.what()).find("stricts") != std::string::npos)
        {
            fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red),
                       "Error with 'stricts' parameter: {}\n", e.what());
            exit(EXIT_FAILURE);
        }
        if (stricts)
            exit(EXIT_FAILURE);
    }
}

int print_inputs(struct settings &settings)
{
    fmt::print(fmt::fg(fmt::color::green), "num_photons = {}\n", settings.rendering_params.num_photons);
    fmt::print(fmt::fg(fmt::color::green), "rendering_type = {}\n", settings.rendering_type);

    // scattering parameters
    fmt::print(fmt::fg(fmt::color::green), "sigma_t = {}\n", settings.scattering_params.sigma_t);
    fmt::print(fmt::fg(fmt::color::green), "albedo = {}\n", settings.scattering_params.albedo);
    fmt::print(fmt::fg(fmt::color::green), "g_val = {}\n", settings.scattering_params.g_val);

    // rif (sources)
    if (settings.rif_params)
    {
        fmt::print(fmt::fg(fmt::color::green), "{}", settings.rif_params->to_string());
    }

    // common / film / geometry
    fmt::print(fmt::fg(fmt::color::green), "emitter_gap = {}\n", settings.emitter_gap);
    fmt::print(fmt::fg(fmt::color::green), "sensor_gap = {}\n", settings.sensor_gap);
    fmt::print(fmt::fg(fmt::color::green), "projector_texture = {}\n", settings.projector_texture);
    fmt::print(fmt::fg(fmt::color::green), "use_direct = {}\n", settings.rendering_params.use_direct);
    fmt::print(fmt::fg(fmt::color::green), "use_angular_sampling = {}\n", settings.rendering_params.use_angular_sampling);
    fmt::print(fmt::fg(fmt::color::green), "use_bounce_decomposition = {}\n", settings.use_bounce_decomposition);
    fmt::print(fmt::fg(fmt::color::green), "max_depth = {}\n", settings.rendering_params.max_depth);
    fmt::print(fmt::fg(fmt::color::green), "max_pathlength = {}\n", settings.rendering_params.max_pathlength);

    fmt::print(fmt::fg(fmt::color::green), "total_medium_length = {}\n",
               settings.scene_params.medium_r[0] - settings.scene_params.medium_l[0]);

    fmt::print(fmt::fg(fmt::color::green), "path_length_min = {}\n", settings.film_params.path_length_min);
    fmt::print(fmt::fg(fmt::color::green), "path_length_max = {}\n", settings.film_params.path_length_max);
    fmt::print(fmt::fg(fmt::color::green), "path_length_bins = {}\n", settings.film_params.path_length_bins);
    fmt::print(fmt::fg(fmt::color::green), "spatial_x = {}\n", settings.film_params.spatial_x);
    fmt::print(fmt::fg(fmt::color::green), "spatial_y = {}\n", settings.film_params.spatial_y);

    fmt::print(fmt::fg(fmt::color::green), "half_theta_limit = {}\n", settings.adoc_geometry_params.half_theta_limit);
    fmt::print(fmt::fg(fmt::color::green), "emitter_size = {}\n", settings.adoc_geometry_params.emitter_size);
    fmt::print(fmt::fg(fmt::color::green), "sensor_size = {}\n", settings.adoc_geometry_params.sensor_size);

    fmt::print(fmt::fg(fmt::color::green), "distribution = {}\n", settings.importance_sampling_params.distribution);
    fmt::print(fmt::fg(fmt::color::green), "g_or_kappa = {}\n", settings.importance_sampling_params.g_or_kappa);

    fmt::print(fmt::fg(fmt::color::green), "emitter_distance = {}\n", settings.adoc_geometry_params.emitter_distance);
    fmt::print(fmt::fg(fmt::color::green), "emitter_lens_aperture = {}\n", settings.lens_params.emitter_lens_aperture);
    fmt::print(fmt::fg(fmt::color::green), "emitter_lens_focal_length = {}\n", settings.lens_params.emitter_lens_focal_length);
    fmt::print(fmt::fg(fmt::color::green), "emitter_lens_active = {}\n", settings.lens_params.emitter_lens_active);

    fmt::print(fmt::fg(fmt::color::green), "sensor_distance = {}\n", settings.adoc_geometry_params.sensor_distance);
    fmt::print(fmt::fg(fmt::color::green), "sensor_lens_aperture = {}\n", settings.lens_params.sensor_lens_aperture);
    fmt::print(fmt::fg(fmt::color::green), "sensor_lens_focal_length = {}\n", settings.lens_params.sensor_lens_focal_length);
    fmt::print(fmt::fg(fmt::color::green), "sensor_lens_active = {}\n", settings.lens_params.sensor_lens_active);

    fmt::print(fmt::fg(fmt::color::green), "print_inputs = {}\n", settings.print_inputs);

    return 0;
}


int sanity_checks(struct settings &settings)
{
    assert(settings.adoc_geometry_params.emitter_distance >= 0 &&
           "emitter_distance should be strictly non-zero");

    assert(settings.adoc_geometry_params.sensor_distance >= 0 &&
           "sensor_distance should be strictly non-zero");

    assert(!(settings.lens_params.emitter_lens_active &&
             settings.adoc_geometry_params.emitter_distance < 1e-4) &&
           "lens_active and emitter_distance. emitter_distance should be strictly positive (>1e-4)");

    assert(!(settings.lens_params.sensor_lens_active &&
             settings.adoc_geometry_params.sensor_distance < 1e-4) &&
           "lens_active and sensor_distance. sensor_distance should be strictly positive (>1e-4)");

    assert(settings.emitter_gap >= 0 &&
           settings.emitter_gap <= (settings.scene_params.medium_r[0] - settings.scene_params.medium_l[0]) &&
           "invalid gap between the emitter and the US");

    assert(settings.sensor_gap >= 0 &&
           settings.sensor_gap <= (settings.scene_params.medium_r[0] - settings.scene_params.medium_l[0]) &&
           "invalid gap between the sensor and the US");

    assert((settings.sensor_gap + settings.emitter_gap) <=
           (settings.scene_params.medium_r[0] - settings.scene_params.medium_l[0]) &&
           "sum of sensor and emitter gaps is more than the medium size");
    return 0;
}
