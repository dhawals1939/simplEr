#include <render_settings_parser.h>
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
    struct settings settings;
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

    parse_config(config, settings, stricts);


    if (settings.print_inputs)
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "simplER renderer started with the following parameters:\n");
        print_inputs(settings);
    }

    sanity_checks(settings);

    pfunc::HenyeyGreenstein *phase = new pfunc::HenyeyGreenstein(settings.scattering_params.g_val);

    tvec::Vec3f emitter_lens_origin(settings.scene_params.medium_r[0], FPCONST(0.0), FPCONST(0.0));
    Float EgapEndLocX = emitter_lens_origin.x - settings.emitter_gap;
    tvec::Vec3f sensor_lens_origin(settings.scene_params.medium_l[0], FPCONST(0.0), FPCONST(0.0));
    Float SgapBeginLocX = sensor_lens_origin.x + settings.sensor_gap; // ADI: VERIFY ME

    /*
        * Initialize source parameters.
        */
    const tvec::Vec3f light_origin(settings.scene_params.medium_r[0] + settings.adoc_geometry_params.emitter_distance, FPCONST(0.0), FPCONST(0.0));
    const Float light_angle = 3.14159265358979323846f;
    const tvec::Vec3f light_dir(std::cos(light_angle), std::sin(light_angle),
                                FPCONST(0.0));
    const tvec::Vec2f light_plane(settings.adoc_geometry_params.emitter_size, settings.adoc_geometry_params.emitter_size);
    const Float Li = FPCONST(75000.0);

    /*
        * Initialize camera parameters.
        */
    const tvec::Vec3f view_origin(settings.scene_params.medium_l[0] - settings.adoc_geometry_params.sensor_distance, FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f view_dir(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f view_x(FPCONST(0.0), -FPCONST(1.0), FPCONST(0.0));
    const tvec::Vec2f view_plane(settings.adoc_geometry_params.sensor_size, settings.adoc_geometry_params.sensor_size);
    const tvec::Vec2f pathlength_range(settings.film_params.path_length_min, settings.film_params.path_length_max);

    const tvec::Vec3i view_resolution(settings.film_params.spatial_x, settings.film_params.spatial_y, settings.film_params.path_length_bins);

    /*
        * Initialize rendering parameters.
        */
    const tvec::Vec3f axis_uz(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f axis_ux(FPCONST(0.0), FPCONST(0.0), FPCONST(1.0));
    const tvec::Vec3f p_u(FPCONST(0.0), FPCONST(0.0), FPCONST(0.0));

    const med::Medium medium(settings.scattering_params.sigma_t, settings.scattering_params.albedo, phase);
    scn::Scene<tvec::TVector3> *scene = nullptr;
    if (settings.rendering_type == "rif_sources")
    {
        auto *rif_params = dynamic_cast<rif_sources *>(settings.rif_params.get());
        if (!rif_params)
        throw std::runtime_error("rif_sources expected, but rif_params holds another type");
        
        #if USE_RIF_SOURCES
        scene = new scn::Scene<tvec::TVector3>(ior, settings.scene_params.medium_l, settings.scene_params.medium_r,
                                               light_origin, light_dir, settings.adoc_geometry_params.half_theta_limit, settings.projector_texture, light_plane, Li,
                                               view_origin, view_dir, view_x, view_plane, pathlength_range, settings.use_bounce_decomposition,
                                               settings.importance_sampling_params.distribution, settings.importance_sampling_params.g_or_kappa,
                                               emitter_lens_origin, settings.lens_params.emitter_lens_aperture, settings.lens_params.emitter_lens_focal_length, settings.lens_params.emitter_lens_active,
                                               sensor_lens_origin, settings.lens_params.sensor_lens_aperture, settings.lens_params.sensor_lens_focal_length, settings.lens_params.sensor_lens_active,
                                               rif_params->f_u, rif_params->speed_u, rif_params->n_o, rif_params->n_scaling, rif_params->n_coeff, rif_params->radius, rif_params->center1, rif_params->center2, rif_params->active1, rif_params->active2, rif_params->phase1, rif_params->phase2, rif_params->theta_min, rif_params->theta_max, rif_params->theta_sources, rif_params->trans_z_min, rif_params->trans_z_max, rif_params->trans_z_sources,
                                               axis_uz, axis_ux, p_u, settings.er_stepsize, settings.direct_to_l, settings.rr_weight, settings.precision, EgapEndLocX, SgapBeginLocX, settings.use_initialization_hack
        );
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::blue), "reached here\n");
        #endif
    }
    else if (settings.rendering_type == "rif_analytical")
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::blue), "reached here rif_analytical\n");

        auto *rif_params = dynamic_cast<rif_analytical *>(settings.rif_params.get());
        if (!rif_params)
            throw std::runtime_error("rif_analytical expected, but rif_params holds another type");

        #if USE_RIF_ANALYTICAL
        scene = new scn::Scene<tvec::TVector3>(ior, settings.scene_params.medium_l, settings.scene_params.medium_r,
                                               light_origin, light_dir, settings.adoc_geometry_params.half_theta_limit, settings.projector_texture, light_plane, Li,
                                               view_origin, view_dir, view_x, view_plane, pathlength_range, settings.use_bounce_decomposition,
                                               settings.importance_sampling_params.distribution, settings.importance_sampling_params.g_or_kappa,
                                               emitter_lens_origin, settings.lens_params.emitter_lens_aperture, settings.lens_params.emitter_lens_focal_length, settings.lens_params.emitter_lens_active,
                                               sensor_lens_origin, settings.lens_params.sensor_lens_aperture, settings.lens_params.sensor_lens_focal_length, settings.lens_params.sensor_lens_active,
                                               rif_params->f_u, rif_params->speed_u, rif_params->n_o, rif_params->n_max, rif_params->n_clip, rif_params->phi_min, rif_params->phi_max, rif_params->mode,
                                               axis_uz, axis_ux, p_u, settings.er_stepsize, settings.direct_to_l, settings.rr_weight, settings.precision, EgapEndLocX, SgapBeginLocX, settings.use_initialization_hack
        );
        #endif
    }
    else if (settings.rendering_type == "rif_interpolated")
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::blue), "reached here rif_interpolated\n");

        auto *rif_params = dynamic_cast<rif_interpolated *>(settings.rif_params.get());
        if (!rif_params)
            throw std::runtime_error("rif_interpolated expected, but rif_params holds another type");

        #if USE_RIF_INTERPOLATED
        scene = new scn::Scene<tvec::TVector3>(ior, settings.scene_params.medium_l, settings.scene_params.medium_r,
                                       light_origin, light_dir, settings.adoc_geometry_params.half_theta_limit, settings.projector_texture, light_plane, Li,
                                       view_origin, view_dir, view_x, view_plane, pathlength_range, settings.use_bounce_decomposition,
                                       settings.importance_sampling_params.distribution, settings.importance_sampling_params.g_or_kappa,
                                       emitter_lens_origin, settings.lens_params.emitter_lens_aperture, settings.lens_params.emitter_lens_focal_length, settings.lens_params.emitter_lens_active,
                                       sensor_lens_origin, settings.lens_params.sensor_lens_aperture, settings.lens_params.sensor_lens_focal_length, settings.lens_params.sensor_lens_active,
                                       rif_params->f_u, rif_params->speed_u, rif_params->n_o, rif_params->n_max, rif_params->n_clip, rif_params->phi_min, rif_params->phi_max, rif_params->mode,
                                       axis_uz, axis_ux, p_u, settings.er_stepsize, settings.direct_to_l, settings.rr_weight, settings.precision, EgapEndLocX, SgapBeginLocX, settings.use_initialization_hack,
                                       rif_params->rifgrid_file
                                    );
        #endif
    }

    if (!scene)
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), "Failed to create scene. Unknown rendering type: {}\n", settings.rendering_type);
        return 1;
    }

    photon::Renderer<tvec::TVector3> renderer(settings.rendering_params.max_depth, settings.rendering_params.max_pathlength, settings.rendering_params.use_direct, settings.rendering_params.use_angular_sampling, settings.threads);

    image::SmallImage img(view_resolution.x, view_resolution.y, view_resolution.z);
    renderer.renderImage(img, medium, *scene, settings.rendering_params.num_photons);

    img.writePFM3D(settings.rendering_type + ".pfm3d");

    delete phase;

    return 0;
}


