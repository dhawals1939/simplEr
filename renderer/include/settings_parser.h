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
    Float m_half_theta_limit;
    Float m_emitter_size;
    Float m_sensor_size;
    Float m_emitter_distance;
    Float m_sensor_distance;

    // Vanilla initializer
    adoc_geometry_parameters(
        Float half_theta_limit_ = FPCONST(12.8e-3),
        Float emitter_size_ = FPCONST(0.002),
        Float sensor_size_ = FPCONST(0.002),
        Float emitter_distance_ = FPCONST(0.0),
        Float sensor_distance_ = FPCONST(0.0))
        : m_half_theta_limit(half_theta_limit_),
          m_emitter_size(emitter_size_),
          m_sensor_size(sensor_size_),
          m_emitter_distance(emitter_distance_),
          m_sensor_distance(sensor_distance_) {}

    // AnyMap initializer
    adoc_geometry_parameters(const AnyMap &adoc_geometry_parameters_json);
    ~adoc_geometry_parameters() = default;
    friend std::ostream& operator<<(std::ostream& os, const adoc_geometry_parameters& params) {
        os << fmt::format(
            "adoc_geometry_parameters {{ half_theta_limit: {}, emitter_size: {}, sensor_size: {}, emitter_distance: {}, sensor_distance: {} }}",
            params.m_half_theta_limit, params.m_emitter_size, params.m_sensor_size, params.m_emitter_distance, params.m_sensor_distance
        );
        return os;
    }
};

// Film parameters
class film_parameters {
public:
    Float m_path_length_min;
    Float m_path_length_max;
    int m_path_length_bins;
    int m_spatial_x;
    int m_spatial_y;

    // Vanilla initializer
    film_parameters(
        Float path_length_min_ = FPCONST(0.0),
        Float path_length_max_ = FPCONST(64.0),
        int path_length_bins_ = 128,
        int spatial_x_ = 128,
        int spatial_y_ = 128)
        : m_path_length_min(path_length_min_),
          m_path_length_max(path_length_max_),
          m_path_length_bins(path_length_bins_),
          m_spatial_x(spatial_x_),
          m_spatial_y(spatial_y_) {}

    // AnyMap initializer
    film_parameters(const AnyMap &film_parameters_json);
    ~film_parameters() = default;
    friend std::ostream& operator<<(std::ostream& os, const film_parameters& params) {
        os << fmt::format(
            "film_parameters {{ path_length_min: {}, path_length_max: {}, path_length_bins: {}, spatial_x: {}, spatial_y: {} }}",
            params.m_path_length_min, params.m_path_length_max, params.m_path_length_bins, params.m_spatial_x, params.m_spatial_y
        );
        return os;
    }
};

// Scattering parameters
class scattering_parameters
{
public:
    Float m_sigma_t;
    Float m_albedo;
    Float m_g_val;

    // Vanilla initializer
    scattering_parameters(
        Float sigma_t_ = FPCONST(0.0),
        Float albedo_ = FPCONST(1.0),
        Float g_val_ = FPCONST(0.0))
        : m_sigma_t(sigma_t_), m_albedo(albedo_), m_g_val(g_val_) {}

    // AnyMap initializer
    scattering_parameters(const AnyMap &scattering_parameters_json);
    ~scattering_parameters() = default;
    friend std::ostream& operator<<(std::ostream& os, const scattering_parameters& params) {
        os << fmt::format(
            "scattering_parameters {{ sigma_t: {}, albedo: {}, g_val: {} }}",
            params.m_sigma_t, params.m_albedo, params.m_g_val
        );
        return os;
    }
};

// Rendering parameters
class rendering_parameters
{
public:
    int64 m_num_photons;
    int m_max_depth;
    Float m_max_pathlength;
    bool m_use_direct;
    bool m_use_angular_sampling;

    // Vanilla initializer
    rendering_parameters(
        int64 num_photons_ = 500L,
        int max_depth_ = -1,
        Float max_pathlength_ = -1,
        bool use_direct_ = false,
        bool use_angular_sampling_ = true)
        : m_num_photons(num_photons_),
          m_max_depth(max_depth_),
          m_max_pathlength(max_pathlength_),
          m_use_direct(use_direct_),
          m_use_angular_sampling(use_angular_sampling_) {}

    // AnyMap initializer
    rendering_parameters(const AnyMap &rendering_parameters_json);
    ~rendering_parameters() = default;
    friend std::ostream& operator<<(std::ostream& os, const rendering_parameters& params) {
        os << fmt::format(
            "rendering_parameters {{ num_photons: {}, max_depth: {}, max_pathlength: {}, use_direct: {}, use_angular_sampling: {} }}",
            params.m_num_photons, params.m_max_depth, params.m_max_pathlength, params.m_use_direct, params.m_use_angular_sampling
        );
        return os;
    }
};

// Scene parameters
class scene_parameters
{
public:
    tvec::Vec3f m_medium_l;
    tvec::Vec3f m_medium_r;

    // Vanilla initializer
    scene_parameters(
        tvec::Vec3f medium_l_ = tvec::Vec3f(-FPCONST(.015), -FPCONST(5.0), -FPCONST(5.0)),
        tvec::Vec3f medium_r_ = tvec::Vec3f(FPCONST(.015), FPCONST(5.0), FPCONST(5.0)))
        : m_medium_l(medium_l_), m_medium_r(medium_r_) {}

    // AnyMap initializer
    scene_parameters(const AnyMap &scene_parameters_json);
    ~scene_parameters() = default;
    friend std::ostream& operator<<(std::ostream& os, const scene_parameters& params) {
        os << fmt::format(
            "scene_parameters {{ medium_l: [{}, {}, {}], medium_r: [{}, {}, {}] }}",
            params.m_medium_l.x, params.m_medium_l.y, params.m_medium_l.z,
            params.m_medium_r.x, params.m_medium_r.y, params.m_medium_r.z
        );
        return os;
    }
};

// Importance sampling parameters
class importance_sampling_parameters {
public:
    std::string m_distribution;
    Float m_g_or_kappa;

    // Vanilla initializer
    importance_sampling_parameters(
        std::string distribution_ = "vmf",
        Float g_or_kappa_ = 4)
        : m_distribution(distribution_), m_g_or_kappa(g_or_kappa_) {}

    // AnyMap initializer
    importance_sampling_parameters(const AnyMap &importance_sampling_parameters_json);
    ~importance_sampling_parameters() = default;
    friend std::ostream& operator<<(std::ostream& os, const importance_sampling_parameters& params) {
        os << fmt::format(
            "importance_sampling_parameters {{ distribution: {}, g_or_kappa: {} }}",
            params.m_distribution, params.m_g_or_kappa
        );
        return os;
    }
};

// Lens parameters
class lens_parameters {
public:
    Float m_emitter_lens_aperture;
    Float m_emitter_lens_focal_length;
    bool m_emitter_lens_active;
    Float m_sensor_lens_aperture;
    Float m_sensor_lens_focal_length;
    bool m_sensor_lens_active;

    // Vanilla initializer
    lens_parameters(
        Float emitter_lens_aperture_ = .015,
        Float emitter_lens_focal_length_ = .015,
        bool emitter_lens_active_ = false,
        Float sensor_lens_aperture_ = .015,
        Float sensor_lens_focal_length_ = .015,
        bool sensor_lens_active_ = false)
        : m_emitter_lens_aperture(emitter_lens_aperture_),
          m_emitter_lens_focal_length(emitter_lens_focal_length_),
          m_emitter_lens_active(emitter_lens_active_),
          m_sensor_lens_aperture(sensor_lens_aperture_),
          m_sensor_lens_focal_length(sensor_lens_focal_length_),
          m_sensor_lens_active(sensor_lens_active_) {}

    // AnyMap initializer
    lens_parameters(const AnyMap &lens_parameters_json);
    ~lens_parameters() = default;
    friend std::ostream& operator<<(std::ostream& os, const lens_parameters& params) {
        os << fmt::format(
            "lens_parameters {{ emitter_lens_aperture: {}, emitter_lens_focal_length: {}, emitter_lens_active: {}, sensor_lens_aperture: {}, sensor_lens_focal_length: {}, sensor_lens_active: {} }}",
            params.m_emitter_lens_aperture, params.m_emitter_lens_focal_length, params.m_emitter_lens_active,
            params.m_sensor_lens_aperture, params.m_sensor_lens_focal_length, params.m_sensor_lens_active
        );
        return os;
    }
};


class execution_parameters
{
public:
    // Vanilla initializer: accepts all parameters and initializes execution_parameters
    execution_parameters(
        bool stricts = false,
        const std::string& rendering_type = "analytic_rif",
        const std::string& output_file_name = "",
        int threads = -1,
        const adoc_geometry_parameters& adoc_geometry_params = adoc_geometry_parameters(),
        const scattering_parameters& scattering_params = scattering_parameters(),
        const rendering_parameters& rendering_params = rendering_parameters(),
        const scene_parameters& scene_params = scene_parameters(),
        const film_parameters& film_params = film_parameters(),
        const importance_sampling_parameters& importance_sampling_params = importance_sampling_parameters(),
        const lens_parameters& lens_params = lens_parameters(),
        std::unique_ptr<rif> rif_params = nullptr,
        Float emitter_gap = .0,
        Float sensor_gap = .0,
        bool use_bounce_decomposition = true,
        bool print_inputs = true,
        const std::string& projector_texture = "/home/dhawals1939/repos/simplER/renderer/images/White.pfm"
    )
        : m_stricts(stricts),
          m_rendering_type(rendering_type),
          m_output_file_name(output_file_name),
          m_threads(threads),
          m_adoc_geometry_params(adoc_geometry_params),
          m_scattering_params(scattering_params),
          m_rendering_params(rendering_params),
          m_scene_params(scene_params),
          m_film_params(film_params),
          m_importance_sampling_params(importance_sampling_params),
          m_lens_params(lens_params),
          m_rif_params(std::move(rif_params)),
          m_emitter_gap(emitter_gap),
          m_sensor_gap(sensor_gap),
          m_use_bounce_decomposition(use_bounce_decomposition),
          m_print_inputs(print_inputs),
          m_projector_texture(projector_texture)
    {
        this->sanity_checks(*this);
    }

    // Destructor: destroys child classes as well
    ~execution_parameters() = default;
    execution_parameters(const AnyMap &settings_json);
    friend std::ostream& operator<<(std::ostream& os, const execution_parameters& s) {
        os << fmt::format(
            "execution_parameters {{ rendering_type: {}, output_file_name: {}, threads: {}, emitter_gap: {}, sensor_gap: {}, use_bounce_decomposition: {}, print_inputs: {}, projector_texture: {} }}\n",
            s.m_rendering_type, s.m_output_file_name, s.m_threads, s.m_emitter_gap, s.m_sensor_gap, s.m_use_bounce_decomposition, s.m_print_inputs, s.m_projector_texture
        );
        os << s.m_adoc_geometry_params << "\n";
        os << s.m_scattering_params << "\n";
        os << s.m_rendering_params << "\n";
        os << s.m_scene_params << "\n";
        os << s.m_film_params << "\n";
        os << s.m_importance_sampling_params << "\n";
        os << s.m_lens_params << "\n";
        if (s.m_rif_params) {
            if (s.m_rendering_type == "rif_sources") {
                os << *static_cast<rif_sources *>(s.m_rif_params.get()) << "\n";
            } else if (s.m_rendering_type == "rif_analytical") {
                os << *static_cast<rif_analytical *>(s.m_rif_params.get()) << "\n";
            } else if (s.m_rendering_type == "rif_interpolated") {
                os << *static_cast<rif_interpolated *>(s.m_rif_params.get()) << "\n";
            }
        }
        return os;
    }
    int sanity_checks(execution_parameters &settings);
    //member variables
    // Output and system parameters
    bool m_stricts = false; // strict parsing of input
    std::string m_rendering_type = "analytic_rif";
    std::string m_output_file_name = "";
    int m_threads = -1; // default number of threads

    // adoc geometry parameters
    adoc_geometry_parameters m_adoc_geometry_params;
    // scattering parameters
    scattering_parameters m_scattering_params;
    // rendering parameters
    rendering_parameters m_rendering_params;
    // scene parameters
    scene_parameters m_scene_params;
    // film parameters
    film_parameters m_film_params;
    // importance sampling parameters
    importance_sampling_parameters m_importance_sampling_params;
    // lens parameters
    lens_parameters m_lens_params;
    // rif parameters
    std::unique_ptr<rif> m_rif_params; // parent class handle

    // Extended rendering parameters
    Float m_emitter_gap = .0; // distance before US activation (from emitter)
    Float m_sensor_gap = .0;  // distance before US activation (towards sensor)

    bool m_use_bounce_decomposition = true; // true = bounce decomposition, false = transient
    bool m_print_inputs = true;
    std::string m_projector_texture = "/home/dhawals1939/repos/simplER/renderer/images/White.pfm";
};