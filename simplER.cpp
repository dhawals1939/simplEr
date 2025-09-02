#include <renderer.h>
#include <util.h>
#include <sampler.h>
#include <vmf.h>
#include <regex>
#include <sstream>
#include <image.h>
#include <vector>
#include <json_parser.h>
#include <fmt/core.h>



/* ADITHYA: Known issues/inconsistencies to be fixed
 * 1. MaxPathLength and PathLengthRange's maxPathLength are inconsistent !!
 * 2. Timing information for the ER is not accurate
 * 3. IOR and n_o are inconsistent
 * 4. Differential rendering part is completely broken
 * 5. 2D rendering is broken
 */

/* tokenize similar to mitsuba, to make input parsing more intuititive */
std::vector<std::string> tokenize(const std::string &string, const std::string &delim)
{
    std::string::size_type lastPos = string.find_first_not_of(delim, 0);
    std::string::size_type pos = string.find_first_of(delim, lastPos);
    std::vector<std::string> tokens;

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        tokens.push_back(string.substr(lastPos, pos - lastPos));
        lastPos = string.find_first_not_of(delim, pos);
        pos = string.find_first_of(delim, lastPos);
    }

    return tokens;
}

int main(int argc, char **argv)
{
    #if USE_RIF_SPLINE
    #if USE_RIF_FUS
        static_assert(false, "Cannot use both spline_RIF and USE_RIF_FUS");
    #endif
    #endif

    /*
     * output file prefix
     */
    std::string outFilePrefix = "USOCTRendering";

    /*
     * System parameters
     */
    int threads = -1; // Default
    /*
     * film parameters
     */
    Float pathLengthMin = FPCONST(0.0);  // This is for path length binning
    Float pathLengthMax = FPCONST(64.0); // This is for path length binning
    int pathLengthBins = 128;
    int spatialX = 128; // X resolution of the film
    int spatialY = 128; // Y resolution of the film

    /*
     * adhoc parameters -- Should be assigned to a better block. Very hacky now.
     */
    Float halfThetaLimit = FPCONST(12.8e-3);
    Float emitter_size = FPCONST(0.002); // size of emitter (square shaped)
    Float sensor_size = FPCONST(0.002);  // size of sensor (square shaped)
    Float emitter_distance = FPCONST(0.0);
    Float sensor_distance = FPCONST(0.0);

    /*
     * Initialize scattering parameters.
     */
    Float sigmaT = FPCONST(0.0);
    Float albedo = FPCONST(1.0);
    Float gVal = FPCONST(0.0);

    /*
     * Initialize scene parameters.
     */
    Float ior = FPCONST(1.3333);
    tvec::Vec3f mediumL(-FPCONST(.015), -FPCONST(5.0), -FPCONST(5.0));
    tvec::Vec3f mediumR(FPCONST(.015), FPCONST(5.0), FPCONST(5.0));

    /*
     * Initialize rendering parameters.
     */
    int64 numPhotons = 500L;
    int maxDepth = -1;
    Float maxPathlength = -1;
    bool useDirect = false;
    bool useAngularSampling = true;

    /*
     * Initialize final path importance sampling parameters.
     */
    std::string distribution = "vmf"; // options are vmf, hg, uniform, none
    Float gOrKappa = 4;

    /*
     * Initialize lens parameters.
     */
    Float emitter_lens_aperture = .015;
    Float emitter_lens_focal_length = .015;
    bool emitter_lens_active = false;

    Float sensor_lens_aperture = .015;
    Float sensor_lens_focal_length = .015;
    bool sensor_lens_active = false;

    bool printInputs = true;

    /*
     * Initialize US parameters
     */
    #if USE_RIF_FUS
    Float f_u = 5 * 1e6;
    Float speed_u = 1500;
    Float n_o = 1.3333;
    Float n_scaling = 0.05e-3;
    Float n_coeff = 1;
    Float radius = 2 * 25.4e-3;
    tvec::Vec3f center1 = {-radius, 0., 0.};
    tvec::Vec3f center2 = {-radius, 0., 0.};
    bool active1 = true;
    bool active2 = true;
    Float phase1 = 0;
    Float phase2 = 0;
    Float chordlength = 0.5 * 25.4e-3;
    Float theta_min = -asin(chordlength / (2 * radius)); // computed from chordlength and radius after parsing the input file
    Float theta_max = asin(chordlength / (2 * radius));  // computed from chordlength and radius after parsing the input file
    int theta_sources = 100;
    Float trans_z_min = -chordlength / 2; // computed from chordlength after parsing the input file
    Float trans_z_max = chordlength / 2;  // computed from chordlength after parsing the input file
    int trans_z_sources = 501;
    #else
    Float f_u = 848 * 1e3;
    Float speed_u = 1500;
    Float n_o = 1.3333;
    Float n_max = 1e-3;
    Float n_clip = 1e-3;
    Float phi_min = M_PI / 2;
    Float phi_max = M_PI / 2;
    int mode = 0;
    #endif

    Float emitter_gap = .0; // Gap is the distance till which US is not ON (from emitter) and only scattering medium is present. Building this feature for Matteo
    Float sensor_gap = .0;  // distance from the transducer to medium boundary (towards sensor) till which only scattering medium is present and no US . Building this feature for Matteo
    Float er_stepsize = 1e-3;
    int precision = 8;                 // Number of dec. precision bits till which we accurately make er_step either because the sampled distances are not an integral multiple of the er_stepsize or because the boundary is hit before.
    Float directTol = 1e-5;            // 10 um units
    bool useInitializationHack = true; // initializationHack forces the direction connections to start from the line connecting both the end points
    Float rrWeight = 1e-2;             // only one in hundred survives second path call

    bool useBounceDecomposition = true; // true is bounce decomposition and false is transient.
    /*
     * Spline approximation, spline parameters
     */
    #if USE_RIF_SPLINE
    //  Float xmin[] = {-0.01, -0.01};
    //  Float xmax[] = { 0.01,  0.01};
    //  int N[] = {21, 21};
    std::string rifgridFile = "us";
    #endif
    /*
     * Projector texture
     */
    std::string projectorTexture("/home/dhawals1939/repos/simplER/renderer/images/White.pfm");

    bool stricts = false;
    std::string configFile = (argc > 1) ? argv[1] : "config.json";
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::green), "Using config file: {}\n", configFile);
    std::ifstream file(configFile);
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
    try
    {    
        stricts = get_exact<bool>(config, "stricts");
        threads = get_num<int>(config, "threads");
        precision = get_num<int>(config, "precision");
        numPhotons = get_num<int>(config, "numPhotons");
        outFilePrefix = get_exact<std::string>(config, "outFilePrefix");

        // Append current date and time to outFilePrefix in format YYYY_MM_DD_HR_MIN
        {
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            char datetime[32];
            std::strftime(datetime, sizeof(datetime), "%Y_%m_%d_%H_%M", &tm);
            outFilePrefix += "_";
            outFilePrefix += datetime;
        }

        // numeric fields — read as double (or float if you prefer)
        sigmaT = get_num<Float>(config, "sigmaT");
        albedo = get_num<Float>(config, "albedo");
        gVal = get_num<Float>(config, "gVal");

    #if USE_RIF_FUS
        f_u = get_num<Float>(config, "f_u");
        speed_u = get_num<Float>(config, "speed_u");
        n_o = get_num<Float>(config, "n_o");
        n_scaling = get_num<Float>(config, "n_scaling");
        n_coeff = get_num<Float>(config, "n_coeff");
        radius = get_num<Float>(config, "radius");
        {
            auto v = get_num_array<Float>(config, "center1");
            for (int i = 0; i < 3; ++i)
                center1[i] = v.at(i);
        }
        {
            auto v = get_num_array<Float>(config, "center2");
            for (int i = 0; i < 3; ++i)
                center2[i] = v.at(i);
        }
        active1 = get_exact<bool>(config, "active1");
        active2 = get_exact<bool>(config, "active2");
        phase1 = get_num<Float>(config, "phase1");
        phase2 = get_num<Float>(config, "phase2");
        chordlength = get_num<Float>(config, "chordlength");
        theta_sources = get_num<int>(config, "theta_sources");
        trans_z_sources = get_num<int>(config, "trans_z_sources");
    #else
        f_u = get_num<Float>(config, "f_u");
        speed_u = get_num<Float>(config, "speed_u");
        n_o = get_num<Float>(config, "n_o");
        n_max = get_num<Float>(config, "n_max");
        n_clip = get_num<Float>(config, "n_clip");
        phi_min = get_num<Float>(config, "phi_min");
        phi_max = get_num<Float>(config, "phi_max");
        mode = get_num<int>(config, "mode");
    #endif

        emitter_gap = get_num<Float>(config, "emitter_gap");
        sensor_gap = get_num<Float>(config, "sensor_gap");
        er_stepsize = get_num<Float>(config, "er_stepsize");
        directTol = get_num<Float>(config, "directTol");
        useInitializationHack = get_exact<bool>(config, "useInitializationHack");
        rrWeight = get_num<Float>(config, "rrWeight");
        projectorTexture = get_exact<std::string>(config, "projectorTexture");
        useDirect = get_exact<bool>(config, "useDirect");
        useAngularSampling = get_exact<bool>(config, "useAngularSampling");
        useBounceDecomposition = get_exact<bool>(config, "useBounceDecomposition");
        maxDepth = get_num<int>(config, "maxDepth");
        maxPathlength = get_num<Float>(config, "maxPathlength");
        pathLengthMin = get_num<Float>(config, "pathLengthMin");
        pathLengthMax = get_num<Float>(config, "pathLengthMax");
        pathLengthBins = get_num<int>(config, "pathLengthBins");
        spatialX = get_num<int>(config, "spatialX");
        spatialY = get_num<int>(config, "spatialY");
        halfThetaLimit = get_num<Float>(config, "halfThetaLimit");
        emitter_size = get_num<Float>(config, "emitter_size");
        sensor_size = get_num<Float>(config, "sensor_size");
        mediumL[0] = get_num<Float>(config, "mediumLx");
        mediumR[0] = get_num<Float>(config, "mediumRx");
        distribution = get_exact<std::string>(config, "distribution");
        gOrKappa = get_num<Float>(config, "gOrKappa");
        emitter_distance = get_num<Float>(config, "emitter_distance");
        emitter_lens_aperture = get_num<Float>(config, "emitter_lens_aperture");
        emitter_lens_focal_length = get_num<Float>(config, "emitter_lens_focal_length");
        emitter_lens_active = get_exact<bool>(config, "emitter_lens_active");
        sensor_distance = get_num<Float>(config, "sensor_distance");
        sensor_lens_aperture = get_num<Float>(config, "sensor_lens_aperture");
        sensor_lens_focal_length = get_num<Float>(config, "sensor_lens_focal_length");
        sensor_lens_active = get_exact<bool>(config, "sensor_lens_active");
        printInputs = get_exact<bool>(config, "printInputs");
        
        #if USE_RIF_SPLINE
        rifgridFile = get_exact<std::string>(config, "rifgridFile");
        #endif
    }
    catch(const std::exception& e)
    {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), "Error occurred while starting simplER renderer: {}\n", e.what());
        if(stricts)
            exit(EXIT_FAILURE);
    }

    std::cout<< "simplER renderer started with the following parameters:\n";

    if (printInputs)
    {
        fmt::print(fmt::fg(fmt::color::green), "numPhotons = {}\n", numPhotons);
        fmt::print(fmt::fg(fmt::color::green), "outFilePrefix = {}\n", outFilePrefix);
        fmt::print(fmt::fg(fmt::color::green), "sigmaT = {}\n", sigmaT);
        fmt::print(fmt::fg(fmt::color::green), "albedo = {}\n", albedo);
        fmt::print(fmt::fg(fmt::color::green), "gVal = {}\n", gVal);
        #if USE_RIF_FUS
        fmt::print(fmt::fg(fmt::color::green), "f_u = {}\n", f_u);
        fmt::print(fmt::fg(fmt::color::green), "speed_u = {}\n", speed_u);
        fmt::print(fmt::fg(fmt::color::green), "n_o = {}\n", n_o);
        fmt::print(fmt::fg(fmt::color::green), "n_scaling = {}\n", n_scaling);
        fmt::print(fmt::fg(fmt::color::green), "n_coeff = {}\n", n_coeff);
        fmt::print(fmt::fg(fmt::color::green), "radius = {}\n", radius);
        fmt::print(fmt::fg(fmt::color::green), "center1 = ({}, {}, {})\n", center1[0], center1[1], center1[2]);
        fmt::print(fmt::fg(fmt::color::green), "center2 = ({}, {}, {})\n", center2[0], center2[1], center2[2]);
        fmt::print(fmt::fg(fmt::color::green), "active1 = {}\n", active1);
        fmt::print(fmt::fg(fmt::color::green), "active2 = {}\n", active2);
        fmt::print(fmt::fg(fmt::color::green), "phase1 = {}\n", phase1);
        fmt::print(fmt::fg(fmt::color::green), "phase2 = {}\n", phase2);
        fmt::print(fmt::fg(fmt::color::green), "chordlength = {}\n", chordlength);
        fmt::print(fmt::fg(fmt::color::green), "theta_sources = {}\n", theta_sources);
        fmt::print(fmt::fg(fmt::color::green), "trans_z_sources = {}\n", trans_z_sources);
        #else
        fmt::print(fmt::fg(fmt::color::green), "f_u = {}\n", f_u);
        fmt::print(fmt::fg(fmt::color::green), "speed_u = {}\n", speed_u);
        fmt::print(fmt::fg(fmt::color::green), "n_o = {}\n", n_o);
        fmt::print(fmt::fg(fmt::color::green), "n_max = {}\n", n_max);
        fmt::print(fmt::fg(fmt::color::green), "n_clip = {}\n", n_clip);
        fmt::print(fmt::fg(fmt::color::green), "phi_min = {}\n", phi_min);
        fmt::print(fmt::fg(fmt::color::green), "phi_max = {}\n", phi_max);
        fmt::print(fmt::fg(fmt::color::green), "mode = {}\n", mode);
        #endif
        fmt::print(fmt::fg(fmt::color::green), "emitter_gap = {}\n", emitter_gap);
        fmt::print(fmt::fg(fmt::color::green), "sensor_gap = {}\n", sensor_gap);
        fmt::print(fmt::fg(fmt::color::green), "projectorTexture = {}\n", projectorTexture);
        fmt::print(fmt::fg(fmt::color::green), "useDirect = {}\n", useDirect);
        fmt::print(fmt::fg(fmt::color::green), "useAngularSampling = {}\n", useAngularSampling);
        fmt::print(fmt::fg(fmt::color::green), "useBounceDecomposition = {}\n", useBounceDecomposition);
        fmt::print(fmt::fg(fmt::color::green), "maxDepth = {}\n", maxDepth);
        fmt::print(fmt::fg(fmt::color::green), "maxPathlength = {}\n", maxPathlength);
        fmt::print(fmt::fg(fmt::color::green), "Total medium length = {}\n", mediumR[0] - mediumL[0]);
        fmt::print(fmt::fg(fmt::color::green), "pathLengthMin = {}\n", pathLengthMin);
        fmt::print(fmt::fg(fmt::color::green), "pathLengthMax = {}\n", pathLengthMax);
        fmt::print(fmt::fg(fmt::color::green), "pathLengthBins = {}\n", pathLengthBins);
        fmt::print(fmt::fg(fmt::color::green), "spatialX = {}\n", spatialX);
        fmt::print(fmt::fg(fmt::color::green), "spatialY = {}\n", spatialY);
        fmt::print(fmt::fg(fmt::color::green), "halfThetaLimit = {}\n", halfThetaLimit);
        fmt::print(fmt::fg(fmt::color::green), "emitter_size = {}\n", emitter_size);
        fmt::print(fmt::fg(fmt::color::green), "sensor_size = {}\n", sensor_size);
        fmt::print(fmt::fg(fmt::color::green), "distribution = {}\n", distribution);
        fmt::print(fmt::fg(fmt::color::green), "gOrKappa = {}\n", gOrKappa);
        fmt::print(fmt::fg(fmt::color::green), "emitter_distance = {}\n", emitter_distance);
        fmt::print(fmt::fg(fmt::color::green), "emitter_lens_aperture = {}\n", emitter_lens_aperture);
        fmt::print(fmt::fg(fmt::color::green), "emitter_lens_focal_length = {}\n", emitter_lens_focal_length);
        fmt::print(fmt::fg(fmt::color::green), "emitter_lens_active = {}\n", emitter_lens_active);
        fmt::print(fmt::fg(fmt::color::green), "sensor_distance = {}\n", sensor_distance);
        fmt::print(fmt::fg(fmt::color::green), "sensor_lens_aperture = {}\n", sensor_lens_aperture);
        fmt::print(fmt::fg(fmt::color::green), "sensor_lens_focal_length = {}\n", sensor_lens_focal_length);
        fmt::print(fmt::fg(fmt::color::green), "sensor_lens_active = {}\n", sensor_lens_active);
        #if USE_RIF_SPLINE
        fmt::print(fmt::fg(fmt::color::green), "rifgridFile = {}\n", rifgridFile);
        #endif
        fmt::print(fmt::fg(fmt::color::green), "printInputs = {}\n", printInputs);
    }

    // some sanity checks
    {
        if (emitter_distance < 0)
        {
            std::cout << "emitter_distance = " << emitter_distance << " should be strictly non-zero" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (sensor_distance < 0)
        {
            std::cout << "sensor_distance = " << sensor_distance << " should be strictly non-zero" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (emitter_lens_active && emitter_distance < 1e-4)
        {
            std::cout << "lens_active and emitter_distance = " << emitter_distance << ". emitter_distance should be strictly positive (>1e-4) " << std::endl;
            exit(EXIT_FAILURE);
        }
        if (sensor_lens_active && sensor_distance < 1e-4)
        {
            std::cout << "lens_active and sensor_distance = " << sensor_distance << ". sensor_distance should be strictly positive (>1e-4) " << std::endl;
            exit(EXIT_FAILURE);
        }
        if (emitter_gap < 0 || emitter_gap > (mediumR[0] - mediumL[0]))
        {
            std::cout << "invalid gap between the emitter and the US:" << emitter_gap << std::endl;
            exit(EXIT_FAILURE);
        }
        if (sensor_gap < 0 || sensor_gap > (mediumR[0] - mediumL[0]))
        {
            std::cout << "invalid gap between the sensor and the US:" << sensor_gap << std::endl;
            exit(EXIT_FAILURE);
        }
        if ((sensor_gap + emitter_gap) > (mediumR[0] - mediumL[0]))
        {
            std::cout << "sum of sensor and emitter gaps is more than the medium size; sum of gaps is " << (sensor_gap + emitter_gap) << std::endl;
            exit(EXIT_FAILURE);
        }
    #if !USE_RIF_FUS
        if (phi_max < phi_min)
        {
            std::cout << "phi_max must be greater than or equal to phi min" << std::endl;
            exit(EXIT_FAILURE);
        }
    #endif
    }

    #if USE_RIF_FUS
    theta_min = -asin(chordlength / (2 * radius));
    theta_max = asin(chordlength / (2 * radius));
    trans_z_min = -chordlength / 2;
    trans_z_max = chordlength / 2;
    #endif

    pfunc::HenyeyGreenstein *phase = new pfunc::HenyeyGreenstein(gVal);

    tvec::Vec3f emitter_lens_origin(mediumR[0], FPCONST(0.0), FPCONST(0.0));
    Float EgapEndLocX = emitter_lens_origin.x - emitter_gap;
    tvec::Vec3f sensor_lens_origin(mediumL[0], FPCONST(0.0), FPCONST(0.0));
    Float SgapBeginLocX = sensor_lens_origin.x + sensor_gap; // ADI: VERIFY ME

    /*
     * Initialize source parameters.
     */
    const tvec::Vec3f lightOrigin(mediumR[0] + emitter_distance, FPCONST(0.0), FPCONST(0.0));
    const Float lightAngle = FPCONST(M_PI);
    const tvec::Vec3f lightDir(std::cos(lightAngle), std::sin(lightAngle),
                               FPCONST(0.0));
    const tvec::Vec2f lightPlane(emitter_size, emitter_size);
    const Float Li = FPCONST(75000.0);

    /*
     * Initialize camera parameters.
     */
    const tvec::Vec3f viewOrigin(mediumL[0] - sensor_distance, FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f viewDir(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f viewX(FPCONST(0.0), -FPCONST(1.0), FPCONST(0.0));
    const tvec::Vec2f viewPlane(sensor_size, sensor_size);
    const tvec::Vec2f pathlengthRange(pathLengthMin, pathLengthMax);

    const tvec::Vec3i viewReso(spatialX, spatialY, pathLengthBins);

    /*
     * Initialize rendering parameters.
     */
    const tvec::Vec3f axis_uz(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
    const tvec::Vec3f axis_ux(FPCONST(0.0), FPCONST(0.0), FPCONST(1.0));
    const tvec::Vec3f p_u(FPCONST(0.0), FPCONST(0.0), FPCONST(0.0));

    const med::Medium medium(sigmaT, albedo, phase);

    scn::Scene<tvec::TVector3> scene(ior, mediumL, mediumR,
                                     lightOrigin, lightDir, halfThetaLimit, projectorTexture, lightPlane, Li,
                                     viewOrigin, viewDir, viewX, viewPlane, pathlengthRange, useBounceDecomposition,
                                     distribution, gOrKappa,
                                     emitter_lens_origin, emitter_lens_aperture, emitter_lens_focal_length, emitter_lens_active,
                                     sensor_lens_origin, sensor_lens_aperture, sensor_lens_focal_length, sensor_lens_active,
                                     #if USE_RIF_FUS
                                     f_u, speed_u, n_o, n_scaling, n_coeff, radius, center1, center2, active1, active2, phase1, phase2, theta_min, theta_max, theta_sources, trans_z_min, trans_z_max, trans_z_sources,
                                     #else
                                     f_u, speed_u, n_o, n_max, n_clip, phi_min, phi_max, mode,
                                     #endif
                                     axis_uz, axis_ux, p_u, er_stepsize, directTol, rrWeight, precision, EgapEndLocX, SgapBeginLocX, useInitializationHack
                                     #if USE_RIF_SPLINE
                                     ,
                                     rifgridFile
            //                      , xmin, xmax, N
                                     #endif
    );

    photon::Renderer<tvec::TVector3> renderer(maxDepth, maxPathlength, useDirect, useAngularSampling, threads);

    image::SmallImage img(viewReso.x, viewReso.y, viewReso.z);
    renderer.renderImage(img, medium, scene, numPhotons);

    img.writePFM3D(outFilePrefix + ".pfm3d");

    delete phase;

    return 0;
}


