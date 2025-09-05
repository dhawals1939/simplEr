#pragma once
#include <tvector.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ostream.h>
#include <any>
#include <map>
#include <string>
#include <vector>
#include <constants.h>
#include <json_helper.h>
#include <rif.h>

// adoc geometry parameters
class adoc_geometry_parameters
{
public:
    Float half_theta_limit;
    Float emitter_size;
    Float sensor_size;
    Float emitter_distance;
    Float sensor_distance;

    // Vanilla initializer
    adoc_geometry_parameters(
        Float half_theta_limit_ = FPCONST(12.8e-3),
        Float emitter_size_ = FPCONST(0.002),
        Float sensor_size_ = FPCONST(0.002),
        Float emitter_distance_ = FPCONST(0.0),
        Float sensor_distance_ = FPCONST(0.0))
        : half_theta_limit(half_theta_limit_),
          emitter_size(emitter_size_),
          sensor_size(sensor_size_),
          emitter_distance(emitter_distance_),
          sensor_distance(sensor_distance_) {}

    // AnyMap initializer
    adoc_geometry_parameters(const AnyMap &adoc_geometry_parameters_json);

    friend std::ostream& operator<<(std::ostream& os, const adoc_geometry_parameters& params) {
        os << fmt::format(
            "adoc_geometry_parameters {{ half_theta_limit: {}, emitter_size: {}, sensor_size: {}, emitter_distance: {}, sensor_distance: {} }}",
            params.half_theta_limit, params.emitter_size, params.sensor_size, params.emitter_distance, params.sensor_distance
        );
        return os;
    }
};

// Film parameters
class film_parameters {
public:
    Float path_length_min;
    Float path_length_max;
    int path_length_bins;
    int spatial_x;
    int spatial_y;

    // Vanilla initializer
    film_parameters(
        Float path_length_min_ = FPCONST(0.0),
        Float path_length_max_ = FPCONST(64.0),
        int path_length_bins_ = 128,
        int spatial_x_ = 128,
        int spatial_y_ = 128)
        : path_length_min(path_length_min_),
          path_length_max(path_length_max_),
          path_length_bins(path_length_bins_),
          spatial_x(spatial_x_),
          spatial_y(spatial_y_) {}

    // AnyMap initializer
    film_parameters(const AnyMap &film_parameters_json);

    friend std::ostream& operator<<(std::ostream& os, const film_parameters& params) {
        os << fmt::format(
            "film_parameters {{ path_length_min: {}, path_length_max: {}, path_length_bins: {}, spatial_x: {}, spatial_y: {} }}",
            params.path_length_min, params.path_length_max, params.path_length_bins, params.spatial_x, params.spatial_y
        );
        return os;
    }
};

// Scattering parameters
class scattering_parameters
{
public:
    Float sigma_t;
    Float albedo;
    Float g_val;

    // Vanilla initializer
    scattering_parameters(
        Float sigma_t_ = FPCONST(0.0),
        Float albedo_ = FPCONST(1.0),
        Float g_val_ = FPCONST(0.0))
        : sigma_t(sigma_t_), albedo(albedo_), g_val(g_val_) {}

    // AnyMap initializer
    scattering_parameters(const AnyMap &scattering_parameters_json);

    friend std::ostream& operator<<(std::ostream& os, const scattering_parameters& params) {
        os << fmt::format(
            "scattering_parameters {{ sigma_t: {}, albedo: {}, g_val: {} }}",
            params.sigma_t, params.albedo, params.g_val
        );
        return os;
    }
};

// Rendering parameters
class rendering_parameters
{
public:
    int64 num_photons;
    int max_depth;
    Float max_pathlength;
    bool use_direct;
    bool use_angular_sampling;

    // Vanilla initializer
    rendering_parameters(
        int64 num_photons_ = 500L,
        int max_depth_ = -1,
        Float max_pathlength_ = -1,
        bool use_direct_ = false,
        bool use_angular_sampling_ = true)
        : num_photons(num_photons_),
          max_depth(max_depth_),
          max_pathlength(max_pathlength_),
          use_direct(use_direct_),
          use_angular_sampling(use_angular_sampling_) {}

    // AnyMap initializer
    rendering_parameters(const AnyMap &rendering_parameters_json);

    friend std::ostream& operator<<(std::ostream& os, const rendering_parameters& params) {
        os << fmt::format(
            "rendering_parameters {{ num_photons: {}, max_depth: {}, max_pathlength: {}, use_direct: {}, use_angular_sampling: {} }}",
            params.num_photons, params.max_depth, params.max_pathlength, params.use_direct, params.use_angular_sampling
        );
        return os;
    }
};

// Scene parameters
class scene_parameters
{
public:
    tvec::Vec3f medium_l;
    tvec::Vec3f medium_r;

    // Vanilla initializer
    scene_parameters(
        tvec::Vec3f medium_l_ = tvec::Vec3f(-FPCONST(.015), -FPCONST(5.0), -FPCONST(5.0)),
        tvec::Vec3f medium_r_ = tvec::Vec3f(FPCONST(.015), FPCONST(5.0), FPCONST(5.0)))
        : medium_l(medium_l_), medium_r(medium_r_) {}

    // AnyMap initializer
    scene_parameters(const AnyMap &scene_parameters_json);

    friend std::ostream& operator<<(std::ostream& os, const scene_parameters& params) {
        os << fmt::format(
            "scene_parameters {{ medium_l: [{}, {}, {}], medium_r: [{}, {}, {}] }}",
            params.medium_l.x, params.medium_l.y, params.medium_l.z,
            params.medium_r.x, params.medium_r.y, params.medium_r.z
        );
        return os;
    }
};

// Importance sampling parameters
class importance_sampling_parameters {
public:
    std::string distribution;
    Float g_or_kappa;

    // Vanilla initializer
    importance_sampling_parameters(
        std::string distribution_ = "vmf",
        Float g_or_kappa_ = 4)
        : distribution(distribution_), g_or_kappa(g_or_kappa_) {}

    // AnyMap initializer
    importance_sampling_parameters(const AnyMap &importance_sampling_parameters_json);

    friend std::ostream& operator<<(std::ostream& os, const importance_sampling_parameters& params) {
        os << fmt::format(
            "importance_sampling_parameters {{ distribution: {}, g_or_kappa: {} }}",
            params.distribution, params.g_or_kappa
        );
        return os;
    }
};

// Lens parameters
class lens_parameters {
public:
    Float emitter_lens_aperture;
    Float emitter_lens_focal_length;
    bool emitter_lens_active;
    Float sensor_lens_aperture;
    Float sensor_lens_focal_length;
    bool sensor_lens_active;

    // Vanilla initializer
    lens_parameters(
        Float emitter_lens_aperture_ = .015,
        Float emitter_lens_focal_length_ = .015,
        bool emitter_lens_active_ = false,
        Float sensor_lens_aperture_ = .015,
        Float sensor_lens_focal_length_ = .015,
        bool sensor_lens_active_ = false)
        : emitter_lens_aperture(emitter_lens_aperture_),
          emitter_lens_focal_length(emitter_lens_focal_length_),
          emitter_lens_active(emitter_lens_active_),
          sensor_lens_aperture(sensor_lens_aperture_),
          sensor_lens_focal_length(sensor_lens_focal_length_),
          sensor_lens_active(sensor_lens_active_) {}

    // AnyMap initializer
    lens_parameters(const AnyMap &lens_parameters_json);

    friend std::ostream& operator<<(std::ostream& os, const lens_parameters& params) {
        os << fmt::format(
            "lens_parameters {{ emitter_lens_aperture: {}, emitter_lens_focal_length: {}, emitter_lens_active: {}, sensor_lens_aperture: {}, sensor_lens_focal_length: {}, sensor_lens_active: {} }}",
            params.emitter_lens_aperture, params.emitter_lens_focal_length, params.emitter_lens_active,
            params.sensor_lens_aperture, params.sensor_lens_focal_length, params.sensor_lens_active
        );
        return os;
    }
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

void parse_config(const AnyMap &config, struct settings &settings, bool &stricts);
int print_inputs(struct settings &settings);
int sanity_checks(struct settings &settings);