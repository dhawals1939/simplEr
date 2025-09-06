#include <settings_parser.h>
#include <renderer.h>
#include <util.h>
#include <sampler.h>
#include <vmf.h>
#include <regex>
#include <sstream>
#include <image.h>
#include <vector>



/* ADITHYA: Known issues/inconsistencies to be fixed
 * 1. MaxPathLength and PathLengthRange's maxPathLength are inconsistent !!
 * 2. Timing information for the ER is not accurate
 * 3. IOR and n_o are inconsistent
 * 4. Differential rendering part is completely broken
 * 5. 2D rendering is broken
 */

int main(int argc, char **argv)
{
    /*
    * Initialize scene parameters.
    */
    Float ior = FPCONST(1.3333);
    bool stricts = false;
    std::string config_file = (argc > 1) ? argv[1] : "config.json";
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "Using config file: {}\n", config_file);
    std::ifstream file(config_file);
    if (!file.is_open())
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), "Failed to open config.json\n");
        return 1;
    }

    nlohmann::json j;
    file >> j;

    AnyMap config;
    convertJsonToMap(j, config);
    if (config.empty())
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), "Failed to parse config.json\n");
        return 1;
    }

    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::yellow), "Config keys:\n");
    for (const auto& kv : config) {
        fmt::print("  {}\n", kv.first);
    }

    execution_parameters execution_params(config);


    if (execution_params.m_print_inputs)
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "simplER renderer started with the following parameters:\n");
        fmt::print(fmt::fg(fmt::color::green), "{}", fmt::streamed(execution_params));
    }

    pfunc::henyey_greenstein *phase = new pfunc::henyey_greenstein(execution_params.m_scattering_params.m_g_val);

    tvec::Vec3f emitter_lens_origin(execution_params.m_scene_params.m_medium_r[0], FPCONST(0.0), FPCONST(0.0));
    Float EgapEndLocX = emitter_lens_origin.x - execution_params.m_emitter_gap;
    tvec::Vec3f sensor_lens_origin(execution_params.m_scene_params.m_medium_l[0], FPCONST(0.0), FPCONST(0.0));
    Float SgapBeginLocX = sensor_lens_origin.x + execution_params.m_sensor_gap; // ADI: VERIFY ME

    /*
        * Initialize source parameters.
        */
    const tvec::Vec3f light_origin(execution_params.m_scene_params.m_medium_r[0] + execution_params.m_adoc_geometry_params.m_emitter_distance, FPCONST(0.0), FPCONST(0.0));
    const Float light_angle = M_PI;
    const tvec::Vec3f light_dir(std::cos(light_angle), std::sin(light_angle),
                                FPCONST(0.0));
    const tvec::Vec2f light_plane(execution_params.m_adoc_geometry_params.m_emitter_size, execution_params.m_adoc_geometry_params.m_emitter_size);
    const Float Li = FPCONST(75000.0);

    /*
        * Initialize camera parameters.
        */
    const tvec::Vec3f view_origin(execution_params.m_scene_params.m_medium_l[0] - execution_params.m_adoc_geometry_params.m_sensor_distance, FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f view_dir(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f view_x(FPCONST(0.0), -FPCONST(1.0), FPCONST(0.0));
    const tvec::Vec2f view_plane(execution_params.m_adoc_geometry_params.m_sensor_size, execution_params.m_adoc_geometry_params.m_sensor_size);
    const tvec::Vec2f pathlength_range(execution_params.m_film_params.m_path_length_min, execution_params.m_film_params.m_path_length_max);

    const tvec::Vec3i view_resolution(execution_params.m_film_params.m_spatial_x, execution_params.m_film_params.m_spatial_y, execution_params.m_film_params.m_path_length_bins);

    /*
        * Initialize rendering parameters.
        */
    const tvec::Vec3f axis_uz(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f axis_ux(FPCONST(0.0), FPCONST(0.0), FPCONST(1.0));
    const tvec::Vec3f p_u(FPCONST(0.0), FPCONST(0.0), FPCONST(0.0));

    const med::Medium medium(execution_params.m_scattering_params.m_sigma_t, execution_params.m_scattering_params.m_albedo, phase);
    scn::Scene<tvec::TVector3> *scene = nullptr;
    if (execution_params.m_rendering_type == "rif_sources")
    {
        auto *rif_params = dynamic_cast<rif_sources *>(execution_params.m_rif_params.get());
        if (!rif_params)
        throw std::runtime_error("rif_sources expected, but rif_params holds another type");
        
        std::cout<< "rif parms type = " << *rif_params << std::endl;
        #if USE_RIF_SOURCES
        scene = new scn::Scene<tvec::TVector3>(ior, execution_params.m_scene_params.m_medium_l, execution_params.m_scene_params.m_medium_r,
                                               light_origin, light_dir, execution_params.m_adoc_geometry_params.m_half_theta_limit, execution_params.m_projector_texture, light_plane, Li,
                                               view_origin, view_dir, view_x, view_plane, pathlength_range, execution_params.m_use_bounce_decomposition,
                                               execution_params.m_importance_sampling_params.m_distribution, execution_params.m_importance_sampling_params.m_g_or_kappa,
                                               emitter_lens_origin, execution_params.m_lens_params.m_emitter_lens_aperture, execution_params.m_lens_params.m_emitter_lens_focal_length, execution_params.m_lens_params.m_emitter_lens_active,
                                               sensor_lens_origin, execution_params.m_lens_params.m_sensor_lens_aperture, execution_params.m_lens_params.m_sensor_lens_focal_length, execution_params.m_lens_params.m_sensor_lens_active,
                                               rif_params->m_f_u, rif_params->m_speed_u, rif_params->m_n_o, rif_params->m_n_scaling, rif_params->m_n_coeff, rif_params->m_radius, rif_params->m_center1, rif_params->m_center2, rif_params->m_active1, rif_params->m_active2, rif_params->m_phase1, rif_params->m_phase2, rif_params->m_theta_min, rif_params->m_theta_max, rif_params->m_theta_sources, rif_params->m_trans_z_min, rif_params->m_trans_z_max, rif_params->m_trans_z_sources,
                                               axis_uz, axis_ux, p_u, execution_params.m_er_stepsize, execution_params.m_direct_to_l, execution_params.m_rr_weight, execution_params.m_precision, EgapEndLocX, SgapBeginLocX, execution_params.m_use_initialization_hack
        );
        #endif
    }
    else if (execution_params.m_rendering_type == "rif_analytical")
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::blue), "reached here rif_analytical\n");

        auto *rif_params = dynamic_cast<rif_analytical *>(execution_params.m_rif_params.get());
        if (!rif_params)
            throw std::runtime_error("rif_analytical expected, but rif_params holds another type");

        #if USE_RIF_ANALYTICAL
        scene = new scn::Scene<tvec::TVector3>(ior, execution_params.m_scene_params.m_medium_l, execution_params.m_scene_params.m_medium_r,
                                               light_origin, light_dir, execution_params.m_adoc_geometry_params.m_half_theta_limit, execution_params.m_projector_texture, light_plane, Li,
                                               view_origin, view_dir, view_x, view_plane, pathlength_range, execution_params.m_use_bounce_decomposition,
                                               execution_params.m_importance_sampling_params.m_distribution, execution_params.m_importance_sampling_params.m_g_or_kappa,
                                               emitter_lens_origin, execution_params.m_lens_params.m_emitter_lens_aperture, execution_params.m_lens_params.m_emitter_lens_focal_length, execution_params.m_lens_params.m_emitter_lens_active,
                                               sensor_lens_origin, execution_params.m_lens_params.m_sensor_lens_aperture, execution_params.m_lens_params.m_sensor_lens_focal_length, execution_params.m_lens_params.m_sensor_lens_active,
                                               rif_params->m_f_u, rif_params->m_speed_u, rif_params->m_n_o, rif_params->m_n_max, rif_params->m_n_clip, rif_params->m_phi_min, rif_params->m_phi_max, rif_params->m_mode,
                                               axis_uz, axis_ux, p_u, execution_params.m_er_stepsize, execution_params.m_direct_to_l, execution_params.m_rr_weight, execution_params.m_precision, EgapEndLocX, SgapBeginLocX, execution_params.m_use_initialization_hack
        );
        #endif
    }
    else if (execution_params.m_rendering_type == "rif_interpolated")
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::blue), "reached here rif_interpolated\n");

        auto *rif_params = dynamic_cast<rif_interpolated *>(execution_params.m_rif_params.get());
        if (!rif_params)
            throw std::runtime_error("rif_interpolated expected, but rif_params holds another type");

        #if USE_RIF_INTERPOLATED
        scene = new scn::Scene<tvec::TVector3>(ior, execution_params.m_scene_params.m_medium_l, execution_params.m_scene_params.m_medium_r,
                                       light_origin, light_dir, execution_params.m_adoc_geometry_params.m_half_theta_limit, execution_params.m_projector_texture, light_plane, Li,
                                       view_origin, view_dir, view_x, view_plane, pathlength_range, execution_params.m_use_bounce_decomposition,
                                       execution_params.m_importance_sampling_params.m_distribution, execution_params.m_importance_sampling_params.m_g_or_kappa,
                                       emitter_lens_origin, execution_params.m_lens_params.m_emitter_lens_aperture, execution_params.m_lens_params.m_emitter_lens_focal_length, execution_params.m_lens_params.m_emitter_lens_active,
                                       sensor_lens_origin, execution_params.m_lens_params.m_sensor_lens_aperture, execution_params.m_lens_params.m_sensor_lens_focal_length, execution_params.m_lens_params.m_sensor_lens_active,
                                       rif_params->m_f_u, rif_params->m_speed_u, rif_params->m_n_o, rif_params->m_n_max, rif_params->m_n_clip, rif_params->m_phi_min, rif_params->m_phi_max, rif_params->m_mode,
                                       axis_uz, axis_ux, p_u, execution_params.m_er_stepsize, execution_params.m_direct_to_l, execution_params.m_rr_weight, execution_params.m_precision, EgapEndLocX, SgapBeginLocX, execution_params.m_use_initialization_hack,
                                       rif_params->m_rifgrid_file
                                    );
        #endif
    }

    if (!scene)
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), "Failed to create scene. Unknown rendering type: {}\n", execution_params.m_rendering_type);
        return 1;
    }

    photon::Renderer<tvec::TVector3> renderer(execution_params.m_rendering_params.m_max_depth, execution_params.m_rendering_params.m_max_pathlength, execution_params.m_rendering_params.m_use_direct, execution_params.m_rendering_params.m_use_angular_sampling, execution_params.m_threads);

    image::SmallImage img(view_resolution.x, view_resolution.y, view_resolution.z);
    renderer.renderImage(img, medium, *scene, execution_params.m_rendering_params.m_num_photons);

    img.writePFM3D(execution_params.m_rendering_type + ".pfm3d");

    delete phase;

    return 0;
}


