#include <settings_parser.h>

film_parameters::film_parameters(const AnyMap &film_parameters_json)
{
    this->m_path_length_min = get_num_path<Float>(film_parameters_json, "path_length.min");
    this->m_path_length_max = get_num_path<Float>(film_parameters_json, "path_length.max");
    this->m_path_length_bins = get_num_path<int>(film_parameters_json, "path_length.bins");
    this->m_spatial_x = get_num_path<int>(film_parameters_json, "spatial.x");
    this->m_spatial_y = get_num_path<int>(film_parameters_json, "spatial.y");
}

adoc_geometry_parameters::adoc_geometry_parameters(const AnyMap &adoc_geometry_parameters_json)
{
    this->m_half_theta_limit = get_num_path<Float>(adoc_geometry_parameters_json, "half_theta_limit");
    this->m_emitter_size = get_num_path<Float>(adoc_geometry_parameters_json, "emitter_size");
    this->m_sensor_size = get_num_path<Float>(adoc_geometry_parameters_json, "sensor_size");
    this->m_emitter_distance = get_num_path<Float>(adoc_geometry_parameters_json, "emitter_distance");
    this->m_sensor_distance = get_num_path<Float>(adoc_geometry_parameters_json, "sensor_distance");
}

scattering_parameters::scattering_parameters(const AnyMap &scattering_parameters_json)
{
    this->m_sigma_t = get_num_path<Float>(scattering_parameters_json, "sigma_t");
    this->m_albedo = get_num_path<Float>(scattering_parameters_json, "albedo");
    this->m_g_val = get_num_path<Float>(scattering_parameters_json, "g_val");
}

rendering_parameters::rendering_parameters(const AnyMap &rendering_parameters_json)
{
    this->m_num_photons = get_num_path<int64>(rendering_parameters_json, "num_photons");
    this->m_max_depth = get_num_path<int>(rendering_parameters_json, "max_depth");
    this->m_max_pathlength = get_num_path<Float>(rendering_parameters_json, "max_pathlength");
    this->m_use_direct = get_exact_path<bool>(rendering_parameters_json, "use_direct");
    this->m_use_angular_sampling = get_exact_path<bool>(rendering_parameters_json, "use_angular_sampling");
}

importance_sampling_parameters::importance_sampling_parameters(const AnyMap &importance_sampling_parameters_json)
{
    this->m_distribution = get_exact_path<std::string>(importance_sampling_parameters_json, "distribution");
    this->m_g_or_kappa = get_num_path<Float>(importance_sampling_parameters_json, "g_or_kappa");
}

scene_parameters::scene_parameters(const AnyMap &scene_parameters_json)
{
    this->m_medium_l[0] = get_num_path<Float>(scene_parameters_json, "medium_lx");
    this->m_medium_r[0] = get_num_path<Float>(scene_parameters_json, "medium_rx");
}

lens_parameters::lens_parameters(const AnyMap &lens_parameters_json)
{
    this->m_emitter_lens_aperture = get_num_path<Float>(lens_parameters_json, "emitter_lens_aperture");
    this->m_emitter_lens_focal_length = get_num_path<Float>(lens_parameters_json, "emitter_lens_focal_length");
    this->m_emitter_lens_active = get_exact_path<bool>(lens_parameters_json, "emitter_lens_active");
    this->m_sensor_lens_aperture = get_num_path<Float>(lens_parameters_json, "sensor_lens_aperture");
    this->m_sensor_lens_focal_length = get_num_path<Float>(lens_parameters_json, "sensor_lens_focal_length");
    this->m_sensor_lens_active = get_exact_path<bool>(lens_parameters_json, "sensor_lens_active");
}

execution_parameters::execution_parameters(const AnyMap &config)
{
    try
    {
        // ---- flat / top-level ----
        this->m_stricts = get_exact<bool>(config, "stricts");
        this->m_rendering_type = get_exact<std::string>(config, "rendering_type");
        this->m_threads = get_num<int>(config, "threads");
        this->m_precision = get_num<int>(config, "precision");

        this->m_emitter_gap = get_num<Float>(config, "emitter_gap");
        this->m_sensor_gap = get_num<Float>(config, "sensor_gap");
        this->m_er_stepsize = get_num<Float>(config, "er_stepsize");
        this->m_direct_to_l = get_num<Float>(config, "direct_to_l");
        this->m_use_initialization_hack = get_exact<bool>(config, "use_initialization_hack");
        this->m_rr_weight = get_num<Float>(config, "rr_weight");
        this->m_use_bounce_decomposition = get_exact<bool>(config, "use_bounce_decomposition");
        this->m_print_inputs = get_exact<bool>(config, "print_inputs");
        this->m_projector_texture = get_exact<std::string>(config, "projector_texture");

        // ---- validate + timestamp ----
        // create output_file_name
        {
            if (this->m_rendering_type != "rif_analytical" &&
                this->m_rendering_type != "rif_sources" &&
                this->m_rendering_type != "rif_interpolated")
            {
                fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red),
                           "Invalid rendering_type: '{}'. Must be one of "
                           "'rif_analytical', 'rif_sources', or 'rif_interpolated'.\n",
                           this->m_rendering_type);
                exit(EXIT_FAILURE);
            }
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            char datetime[32];
            std::strftime(datetime, sizeof(datetime), "%Y_%m_%d_%H_%M", &tm);
            this->m_output_file_name = this->m_rendering_type + "_";
            this->m_output_file_name += datetime;
        }

        // ---- nested: film_parameters ----
        const auto &film_params_json = get_exact<AnyMap>(config, "film_parameters");
        this->m_film_params = film_parameters(film_params_json);
    
        // ---- nested: adoc_parameters ----
        const auto &adoc_geometry_params_json = get_exact<AnyMap>(config, "adoc_geometry_parameters");
        this->m_adoc_geometry_params = adoc_geometry_parameters(adoc_geometry_params_json);

        // ---- nested: scattering_parameters ----
        const auto &scattering_params_json = get_exact<AnyMap>(config, "scattering_parameters");
        this->m_scattering_params = scattering_parameters(scattering_params_json);

        // ---- nested: rendering_parameters ----
        const auto &rendering_params_json = get_exact<AnyMap>(config, "rendering_parameters");
        this->m_rendering_params = rendering_parameters(rendering_params_json);

        // ---- nested: importance_sampling_parameters ----
        const auto &importance_sampling_params_json = get_exact<AnyMap>(config, "importance_sampling_parameters");
        this->m_importance_sampling_params = importance_sampling_parameters(importance_sampling_params_json);

        // ---- nested: scene_parameters ----
        const auto &scene_params_json = get_exact<AnyMap>(config, "scene_parameters");
        this->m_scene_params = scene_parameters(scene_params_json);

        // ---- nested: lens_parameters ----
        const auto &lens_params_json = get_exact<AnyMap>(config, "lens_parameters");
        this->m_lens_params = lens_parameters(lens_params_json);

        // ---- nested: rif_parameters ----
        const auto &rif_params = get_exact<AnyMap>(config, "rif_parameters");

        if (this->m_rendering_type == "rif_sources")
        {
            this->m_rif_params = std::make_unique<rif_sources>(rif_params); // ctor(AnyMap)
        }
        else if (this->m_rendering_type == "rif_analytical")
        {
            this->m_rif_params = std::make_unique<rif_analytical>(rif_params);
        }
        else if (this->m_rendering_type == "rif_interpolated")
        {
            this->m_rif_params = std::make_unique<rif_interpolated>(rif_params);
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
        if (this->m_stricts)
            exit(EXIT_FAILURE);
    }
    this->sanity_checks(*this);
}


int execution_parameters::sanity_checks(execution_parameters &settings)
{
    assert(settings.m_adoc_geometry_params.m_emitter_distance >= 0 &&
           "emitter_distance should be strictly non-zero");

    assert(settings.m_adoc_geometry_params.m_sensor_distance >= 0 &&
           "sensor_distance should be strictly non-zero");

    assert(!(settings.m_lens_params.m_emitter_lens_active &&
             settings.m_adoc_geometry_params.m_emitter_distance < 1e-4) &&
           "lens_active and emitter_distance. emitter_distance should be strictly positive (>1e-4)");

    assert(!(settings.m_lens_params.m_sensor_lens_active &&
             settings.m_adoc_geometry_params.m_sensor_distance < 1e-4) &&
           "lens_active and sensor_distance. sensor_distance should be strictly positive (>1e-4)");

    assert(settings.m_emitter_gap >= 0 &&
           settings.m_emitter_gap <= (settings.m_scene_params.m_medium_r[0] - settings.m_scene_params.m_medium_l[0]) &&
           "invalid gap between the emitter and the US");

    assert(settings.m_sensor_gap >= 0 &&
           settings.m_sensor_gap <= (settings.m_scene_params.m_medium_r[0] - settings.m_scene_params.m_medium_l[0]) &&
           "invalid gap between the sensor and the US");

    assert((settings.m_sensor_gap + settings.m_emitter_gap) <=
               (settings.m_scene_params.m_medium_r[0] - settings.m_scene_params.m_medium_l[0]) &&
           "sum of sensor and emitter gaps is more than the medium size");
    return 0;
}


