/*
 * renderer_sample.cpp
 *
 *  Created on: Nov 20, 2019
 *      Author: apedired
 */

#include "renderer.h"
#include "util.h"
#include "sampler.h"
#include "vmf.h"
#include <regex>
#include <sstream>

#include <vector>

#ifdef USE_CERES
#include "ceres/ceres.h"
#include "glog/logging.h"


using ceres::CostFunction;
using ceres::SizedCostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
#endif

/* ADITHYA: Known issues/inconsistencies to be fixed
 * 1. MaxPathLength and PathLengthRange's maxPathLength are inconsistent !!
 * 2. Timing information for the ER is not accurate
 * 3. IOR and n_o are inconsistent
 * 4. Differential rendering part is completely broken
 * 5. 2D rendering is broken
 */

/* tokenize similar to mitsuba, to make input parsing more intuititive */
std::vector<std::string> tokenize(const std::string &string, const std::string &delim) {
    std::string::size_type lastPos = string.find_first_not_of(delim, 0);
    std::string::size_type pos = string.find_first_of(delim, lastPos);
    std::vector<std::string> tokens;

    while (std::string::npos != pos || std::string::npos != lastPos) {
        tokens.push_back(string.substr(lastPos, pos - lastPos));
        lastPos = string.find_first_not_of(delim, pos);
        pos = string.find_first_of(delim, lastPos);
    }

    return tokens;
}


int main(int argc, char **argv) {
#ifdef USE_CERES
    google::InitGoogleLogging(argv[0]);
#endif

#ifdef SPLINE_RIF 
#ifdef FUS_RIF
    static_assert(false, "Cannot use both spline_RIF and FUS_RIF");
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
    Float pathLengthMin = FPCONST(0.0); // This is for path length binning
    Float pathLengthMax = FPCONST(64.0); // This is for path length binning
    int pathLengthBins = 128;
    int spatialX = 128; // X resolution of the film 
    int spatialY = 128; // Y resolution of the film

    /*
     * adhoc parameters -- Should be assigned to a better block. Very hacky now. 
     */
    Float halfThetaLimit = FPCONST(12.8e-3);
    Float emitter_size = FPCONST(0.002); // size of emitter (square shaped)
    Float sensor_size = FPCONST(0.002); // size of sensor (square shaped)
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
    tvec::Vec3f mediumR( FPCONST(.015),  FPCONST(5.0),  FPCONST(5.0));

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
    Float emitter_lens_focalLength = .015;
    bool emitter_lens_active = false;

    Float sensor_lens_aperture = .015;
    Float sensor_lens_focalLength = .015;
    bool sensor_lens_active = false;

    bool printInputs = true;

    /*
     * Initialize US parameters
     */
#ifdef FUS_RIF
    Float f_u = 5*1e6;
    Float speed_u = 1500;
    Float n_o = 1.3333;
    Float n_scaling = 0.05e-3;
    int n_coeff = 1;
    Float radius = 2 * 25.4e-3;
    tvec::Vec3f center1 = {-radius, 0., 0.}; 
    tvec::Vec3f center2 = {-radius, 0., 0.};
    bool active1 = true;
    bool active2 = true;
    Float phase1 = 0;
    Float phase2 = 0;
    Float chordlength = 0.5 * 25.4e-3;
    Float theta_min= -asin(chordlength/(2*radius)); //computed from chordlength and radius after parsing the input file
    Float theta_max=  asin(chordlength/(2*radius)); //computed from chordlength and radius after parsing the input file
    int theta_sources = 100;
    Float trans_z_min = -chordlength/2; //computed from chordlength after parsing the input file
    Float trans_z_max =  chordlength/2; //computed from chordlength after parsing the input file
    int trans_z_sources = 501;
#else
    Float f_u = 848*1e3;
    Float speed_u = 1500;
    Float n_o = 1.3333;
    Float n_max = 1e-3;
    Float n_clip= 1e-3;
    Float phi_min = M_PI/2;
    Float phi_max = M_PI/2;
    int mode = 0;
#endif




    Float emitter_gap = .0; // Gap is the distance till which US is not ON (from emitter) and only scattering medium is present. Building this feature for Matteo
    Float sensor_gap = .0; // distance from the transducer to medium boundary (towards sensor) till which only scattering medium is present and no US . Building this feature for Matteo
    Float er_stepsize = 1e-3;
    int precision = 8; // Number of dec. precision bits till which we accurately make er_step either because the sampled distances are not an integral multiple of the er_stepsize or because the boundary is hit before.
    Float directTol = 1e-5; // 10 um units
    bool useInitializationHack = true; // initializationHack forces the direction connections to start from the line connecting both the end points
    Float rrWeight  = 1e-2; // only one in hundred survives second path call

    bool useBounceDecomposition = true; // true is bounce decomposition and false is transient.
    /*
     * Spline approximation, spline parameters
     */
#ifdef SPLINE_RIF
//  Float xmin[] = {-0.01, -0.01};
//  Float xmax[] = { 0.01,  0.01};
//  int N[] = {21, 21};
    std::string rifgridFile = "us";
#endif
    /*
     * Projector texture
     */
    std::string projectorTexture("/home/apedired/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr/renderer/images/White.pfm");

    bool stricts=false;
    bool bthreads=false;
    bool bprecision=false;
    bool bnumPhotons=false;
    bool boutFilePrefix=false;
    bool bsigmaT=false;
    bool balbedo=false;
    bool bgVal=false;
#ifdef FUS_RIF
    bool bf_u=false;
    bool bspeed_u=false;
    bool bn_o=false;
    bool bn_scaling=false;
    bool bn_coeff=false;
    bool bradius=false;
    bool bcenter1=false;
    bool bcenter2=false;
    bool bactive1=false;
    bool bactive2=false;
    bool bphase1=false;
    bool bphase2=false;
    bool bchordlength=false;
    bool btheta_sources=false;
    bool btrans_z_sources=false;
#else
    bool bf_u=false;
    bool bspeed_u=false;
    bool bn_o=false;
    bool bn_max=false;
    bool bn_clip=false;
    bool bphi_min=false;
    bool bphi_max=false;
    bool bmode=false;
#endif
    bool bemitter_gap=false;
    bool bsensor_gap=false;
    bool ber_stepsize=false;
    bool bdirectTol=false;
    bool buseInitializationHack=false;
    bool brrWeight=false;
    bool bprojectorTexture=false;
    bool buseDirect=false;
    bool buseAngularSampling=false;
    bool buseBounceDecomposition=false;
    bool bmaxDepth=false;
    bool bmaxPathlength=false;
    bool bpathLengthMin=false;
    bool bpathLengthMax=false;
    bool bpathLengthBins=false;
    bool bspatialX=false;
    bool bspatialY=false;
    bool bhalfThetaLimit=false;
    bool bemitter_size=false;
    bool bsensor_size=false;
    bool bmediumLx=false;
    bool bmediumRx=false;
    bool bdistribution=false;
    bool bgOrKappa=false;
    bool bemitter_distance=false;
    bool bemitter_lens_aperture=false;
    bool bemitter_lens_focalLength=false;
    bool bemitter_lens_active=false;
    bool bsensor_distance=false;
    bool bsensor_lens_aperture=false;
    bool bsensor_lens_focalLength=false;
    bool bsensor_lens_active=false;
    bool bprintInputs=false;

#ifdef SPLINE_RIF
    bool brifgridFile=false;
#endif

    for(int i = 1; i < argc; i++){
        std::vector<std::string> param = tokenize(argv[i], "=");
        if(param.size() != 2){
            std::cerr << "Input argument " << argv[i] << "should be in the format arg=value" << std::endl;
            return -1;
        }
        if(param[0].compare("stricts")==0){ 
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0) 
                stricts = true;
            else if(param[1].compare("false")==0) 
                stricts = false;
            else{
                std::cerr << "stricts should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
        }else if(param[0].compare("threads")==0){
            bthreads=true;
            threads = stoi(param[1]);
        }else if(param[0].compare("precision")==0){
            bprecision=true;
            precision = stoi(param[1]);
        }else if(param[0].compare("numPhotons")==0){
            bnumPhotons=true;
            numPhotons = stoi(param[1]);
        }else if(param[0].compare("outFilePrefix")==0){
            boutFilePrefix=true;
            outFilePrefix = param[1];
        }else if(param[0].compare("sigmaT")==0){
            bsigmaT=true;
            sigmaT = stof(param[1]);
        }else if(param[0].compare("albedo")==0){
            balbedo=true;
            albedo = stof(param[1]);
        }else if(param[0].compare("gVal")==0){
            bgVal=true;
            gVal = stof(param[1]);
#ifdef FUS_RIF
        }else if(param[0].compare("f_u")==0){
            bf_u=true;
            f_u = stof(param[1]);
        }else if(param[0].compare("speed_u")==0){
            bspeed_u=true;
            speed_u = stof(param[1]);
        }else if(param[0].compare("n_o")==0){
            bn_o=true;
            n_o = stof(param[1]);
        }else if(param[0].compare("n_scaling")==0){
            bn_scaling=true;
            n_scaling = stof(param[1]);
        }else if(param[0].compare("n_coeff")==0){
            bn_coeff=true;
            n_coeff = stoi(param[1]);
        }else if(param[0].compare("radius")==0){
            bradius=true;
            radius = stof(param[1]);
        }else if(param[0].compare("center1")==0){
            bcenter1=true;
            std::regex rgx("\,");
            std::sregex_token_iterator iter(param[1].begin(),param[1].end(), rgx, -1);
            std::sregex_token_iterator end;
            for(int i=0; i<3; i++){
                if(iter == end){
                    std::cout << "Three input coordinates required for center1; crashing \n" << std::endl; return -1;
                }       
                center1[i] = stof(*iter++);
            }
            if(iter != end){
                std::cout << "More than three input coordinates entered for center1; crashing \n" << std::endl; return -1;
            }
        }else if(param[0].compare("center2")==0){
            bcenter2=true;
            std::regex rgx("\,");
            std::sregex_token_iterator iter(param[1].begin(),param[1].end(), rgx, -1);
            std::sregex_token_iterator end;
            for(int i=0; i<3; i++){
                if(iter == end){
                    std::cout << "Three input coordinates required for center2; crashing \n" << std::endl; return -1;
                }       
                center2[i] = stof(*iter++);
            }
            if(iter != end){
                std::cout << "More than three input coordinates entered for center2; crashing \n" << std::endl; return -1;
            }
        }else if(param[0].compare("active1")==0){ 
            bactive1=true;
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0) 
                active1 = true;
            else if(param[1].compare("false")==0) 
                active1 = false;
            else{
                std::cerr << "active1 should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
        }else if(param[0].compare("active2")==0){ 
            bactive2=true;
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0) 
                active2 = true;
            else if(param[1].compare("false")==0) 
                active2 = false;
            else{
                std::cerr << "active2 should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
        }else if(param[0].compare("phase1")==0){
            bphase1=true;
            phase1 = stof(param[1]);
        }else if(param[0].compare("phase2")==0){
            bphase2=true;
            phase2 = stof(param[1]);
        }else if(param[0].compare("chordlength")==0){
            bchordlength=true;
            chordlength = stof(param[1]);
        }else if(param[0].compare("theta_sources")==0){
            btheta_sources=true;
            theta_sources = stoi(param[1]);
        }else if(param[0].compare("trans_z_sources")==0){
            btrans_z_sources=true;
            trans_z_sources = stoi(param[1]);
#else
        }else if(param[0].compare("f_u")==0){
            bf_u=true;
            f_u = stof(param[1]);
        }else if(param[0].compare("speed_u")==0){
            bspeed_u=true;
            speed_u = stof(param[1]);
        }else if(param[0].compare("n_o")==0){
            bn_o=true;
            n_o = stof(param[1]);
        }else if(param[0].compare("n_max")==0){
            bn_max=true;
            n_max = stof(param[1]);
        }else if(param[0].compare("n_clip")==0){
            bn_clip=true;
            n_clip = stof(param[1]);
        }else if(param[0].compare("phi_min")==0){
            bphi_min=true;
            phi_min = stof(param[1]);
        }else if(param[0].compare("phi_max")==0){
            bphi_max=true;
            phi_max = stof(param[1]);
        }else if(param[0].compare("mode")==0){
            bmode=true;
            mode = stoi(param[1]);
#endif
        }else if(param[0].compare("emitter_gap")==0){
            bemitter_gap=true;
            emitter_gap = stof(param[1]);
        }else if(param[0].compare("sensor_gap")==0){
            bsensor_gap=true;
            sensor_gap = stof(param[1]);
        }else if(param[0].compare("er_stepsize")==0){
            ber_stepsize=true;
            er_stepsize = stof(param[1]);
        }else if(param[0].compare("directTol")==0){
            bdirectTol=true;
            directTol = stof(param[1]);
        }else if(param[0].compare("useInitializationHack")==0){ 
            buseInitializationHack=true;
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0) 
                useInitializationHack = true;
            else if(param[1].compare("false")==0) 
                useInitializationHack = false;
            else{
                std::cerr << "useInitializationHack should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
        }else if(param[0].compare("rrWeight")==0){
            brrWeight=true;
            rrWeight = stof(param[1]);
        }else if(param[0].compare("projectorTexture")==0){
            bprojectorTexture=true;
            projectorTexture = param[1];
        }else if(param[0].compare("useDirect")==0){ 
            buseDirect=true;
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0) 
                useDirect = true;
            else if(param[1].compare("false")==0) 
                useDirect = false;
            else{
                std::cerr << "useDirect should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
        }else if(param[0].compare("useAngularSampling")==0){
            buseAngularSampling=true;
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0) 
                useAngularSampling = true;
            else if(param[1].compare("false")==0) 
                useAngularSampling = false;
            else{
                std::cerr << "useAngularSampling should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
        }else if(param[0].compare("useBounceDecomposition")==0){
            buseBounceDecomposition=true;
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0)
                useBounceDecomposition = true;
            else if(param[1].compare("false")==0)
                useBounceDecomposition = false;
            else{
                std::cerr << "useBounceDecompostion should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
        }else if(param[0].compare("maxDepth")==0){
            bmaxDepth=true;
            maxDepth = stoi(param[1]);
        }else if(param[0].compare("maxPathlength")==0){
            bmaxPathlength=true;
            maxPathlength = stof(param[1]);
        }else if(param[0].compare("pathLengthMin")==0){
            bpathLengthMin=true;
            pathLengthMin = stof(param[1]);
        }else if(param[0].compare("pathLengthMax")==0){
            bpathLengthMax=true;
            pathLengthMax = stof(param[1]);
        }else if(param[0].compare("pathLengthBins")==0){
            bpathLengthBins=true;
            pathLengthBins = stoi(param[1]);
        }else if(param[0].compare("spatialX")==0){
            bspatialX=true;
            spatialX = stoi(param[1]);
        }else if(param[0].compare("spatialY")==0){
            bspatialY=true;
            spatialY = stoi(param[1]);
        }else if(param[0].compare("halfThetaLimit")==0){
            bhalfThetaLimit=true;
            halfThetaLimit = stof(param[1]);
        }else if(param[0].compare("emitter_size")==0){
            bemitter_size=true;
            emitter_size = stof(param[1]);
        }else if(param[0].compare("sensor_size")==0){
            bsensor_size=true;
            sensor_size = stof(param[1]);
        }else if(param[0].compare("mediumLx")==0){
            bmediumLx=true;
            mediumL[0] = stof(param[1]);
        }else if(param[0].compare("mediumRx")==0){
            bmediumRx=true;
            mediumR[0] = stof(param[1]);
        }else if(param[0].compare("distribution")==0){
            bdistribution=true;
            distribution = (param[1]);
        }else if(param[0].compare("gOrKappa")==0){
            bgOrKappa=true;
            gOrKappa = stof(param[1]);
        }else if(param[0].compare("emitter_distance")==0){
            bemitter_distance=true;
            emitter_distance = stof(param[1]);
        }else if(param[0].compare("emitter_lens_aperture")==0){
            bemitter_lens_aperture=true;
            emitter_lens_aperture = stof(param[1]);
        }else if(param[0].compare("emitter_lens_focalLength")==0){
            bemitter_lens_focalLength=true;
            emitter_lens_focalLength = stof(param[1]);
        }else if(param[0].compare("emitter_lens_active")==0){
            bemitter_lens_active=true;
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0)
                emitter_lens_active = true;
            else if(param[1].compare("false")==0)
                emitter_lens_active = false;
            else{
                std::cerr << "emitter_lens_active should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
        }else if(param[0].compare("sensor_distance")==0){
            bsensor_distance=true;
            sensor_distance = stof(param[1]);
        }else if(param[0].compare("sensor_lens_aperture")==0){
            bsensor_lens_aperture=true;
            sensor_lens_aperture = stof(param[1]);
        }else if(param[0].compare("sensor_lens_focalLength")==0){
            bsensor_lens_focalLength=true;
            sensor_lens_focalLength = stof(param[1]);
        }else if(param[0].compare("sensor_lens_active")==0){
            bsensor_lens_active=true;
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0) 
                sensor_lens_active = true;
            else if(param[1].compare("false")==0) 
                sensor_lens_active = false;
            else{
                std::cerr << "sensor_lens_active should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
        }else if(param[0].compare("printInputs")==0){ 
            bprintInputs=true;
            transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
            if(param[1].compare("true")==0)
                printInputs = true;
            else if(param[1].compare("false")==0)
                printInputs = false;
            else{
                std::cerr << "printInputs should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
                return -1;
            }
#ifdef SPLINE_RIF
        }else if(param[0].compare("rifgridFile")==0){
            brifgridFile=true;
            rifgridFile = param[1];
#endif
        }else{
            std::cerr << "Unknown variable in the input argument:" << param[0] << std::endl;
            std::cerr << "Should be one of "
                      << "stricts, "
                      << "threads, "
                      << "precision, "
                      << "numPhotons, "
                      << "outFilePrefix, "
                      << "sigmaT, "
                      << "albedo, "
                      << "gVal, "
#ifdef FUS_RIF
                      << "f_u, "
                      << "speed_u, "
                      << "n_o, "
                      << "n_scaling, "
                      << "n_coeff, "
                      << "radius, "
                      << "center1, "
                      << "center2, "
                      << "active1, "
                      << "active2, "
                      << "phase1, "
                      << "phase2, "
                      << "chordlength, "
                      << "theta_min, "
                      << "theta_max, "
                      << "theta_sources, "
                      << "trans_z_min, "
                      << "trans_z_max, "
                      << "trans_z_sources, "
#else
                      << "f_u, "
                      << "speed_u, "
                      << "n_o, "
                      << "n_max, "
                      << "n_clip, "
                      << "phi_min, "
                      << "phi_max, "
#endif
                      << "emitter_gap, "
                      << "sensor_gap, "
                      << "mode, "
                      << "er_stepsize, "
                      << "directTol, "
                      << "rrWeight, "
                      << "projectorTexture, "
                      << "useDirect, "
                      << "useAngularSampling, "
                      << "useBounceDecompostion, "
                      << "maxDepth, "
                      << "maxPathlength, "
                      << "pathLengthMin, "
                      << "pathLengthMax, "
                      << "pathLengthBins, "
                      << "spatialX, "
                      << "spatialY, "
                      << "halfThetaLimit, "
                      << "emitter_size, "
                      << "sensor_size, "
                      << "mediumLx, "
                      << "mediumRx, "
                      << "distribution, "
                      << "gOrKappa, "
                      << "emitter_distance, "
                      << "emitter_lens_aperture, "
                      << "emitter_lens_focalLength, "
                      << "emitter_lens_active, "
                      << "sensor_distance, "
                      << "sensor_lens_aperture, "
                      << "sensor_lens_focalLength, "
                      << "sensor_lens_active, "
#ifdef SPLINE_RIF
                      << "rifgridFile, "
#endif
                      << "printInputs "
                      << std::endl;
            return -1;
        }
    }

    if(stricts){
        if(!bthreads) {std::cout << "threads is not specified " << std::endl;}
        if(!bprecision) {std::cout << "precision is not specified " << std::endl;}
        if(!bnumPhotons) {std::cout << "numPhotons is not specified " << std::endl;}
        if(!boutFilePrefix) {std::cout << "outFilePrefix is not specified " << std::endl;}
        if(!bsigmaT) {std::cout << "sigmaT is not specified " << std::endl;}
        if(!balbedo) {std::cout << "albedo is not specified " << std::endl;}
        if(!bgVal) {std::cout << "gVal is not specified " << std::endl;}
#ifdef FUS_RIF
        if(!bf_u) {std::cout << "f_u is not specified " << std::endl;}
        if(!bspeed_u) {std::cout << "speed_u is not specified " << std::endl;}
        if(!bn_o) {std::cout << "n_o is not specified " << std::endl;}
        if(!bn_scaling) {std::cout << "n_scaling is not specified " << std::endl;}
        if(!bn_coeff) {std::cout << "n_coeff is not specified " << std::endl;}
        if(!bradius) {std::cout << "radius is not specified " << std::endl;}
        if(!bcenter1) {std::cout << "center1 is not specified " << std::endl;}
        if(!bcenter2) {std::cout << "center2 is not specified " << std::endl;}
        if(!bactive1) {std::cout << "active1 is not specified " << std::endl;}
        if(!bactive2) {std::cout << "active2 is not specified " << std::endl;}
        if(!bphase1) {std::cout << "phase1 is not specified " << std::endl;}
        if(!bphase2) {std::cout << "phase2 is not specified " << std::endl;}
        if(!bchordlength) {std::cout << "chordlength is not specified " << std::endl;}
        if(!btheta_sources) {std::cout << "theta_sources is not specified " << std::endl;}
        if(!btrans_z_sources) {std::cout << "trans_z_sources is not specified " << std::endl;}
#else
        if(!bf_u) {std::cout << "f_u is not specified " << std::endl;}
        if(!bspeed_u) {std::cout << "speed_u is not specified " << std::endl;}
        if(!bn_o) {std::cout << "n_o is not specified " << std::endl;}
        if(!bn_max) {std::cout << "n_max is not specified " << std::endl;}
        if(!bn_clip) {std::cout << "n_clip is not specified " << std::endl;}
        if(!bphi_min) {std::cout << "phi_min is not specified " << std::endl;}
        if(!bphi_max) {std::cout << "phi_max is not specified " << std::endl;}
        if(!bmode) {std::cout << "mode is not specified " << std::endl;}
#endif
        if(!bemitter_gap) {std::cout << "emitter_gap is not specified " << std::endl;}
        if(!bsensor_gap) {std::cout << "sensor_gap is not specified " << std::endl;}
        if(!ber_stepsize) {std::cout << "er_stepsize is not specified " << std::endl;}
        if(!buseInitializationHack) {std::cout << "useInitializationHack is not specified " << std::endl;}
        if(!bdirectTol) {std::cout << "directTol is not specified " << std::endl;}
        if(!brrWeight) {std::cout << "rrWeight is not specified " << std::endl;}
        if(!bprojectorTexture) {std::cout << "projectorTexture is not specified " << std::endl;}
        if(!buseDirect) {std::cout << "useDirect is not specified " << std::endl;}
        if(!buseAngularSampling) {std::cout << "useAngularSampling is not specified " << std::endl;}
        if(!buseBounceDecomposition) {std::cout << "useBounceDecomposition is not specified " << std::endl;}
        if(!bmaxDepth) {std::cout << "maxDepth is not specified " << std::endl;}
        if(!bmaxPathlength) {std::cout << "maxPathlength is not specified " << std::endl;}
        if(!bpathLengthMin) {std::cout << "pathLengthMin is not specified " << std::endl;}
        if(!bpathLengthMax) {std::cout << "pathLengthMax is not specified " << std::endl;}
        if(!bpathLengthBins) {std::cout << "pathLengthBins is not specified " << std::endl;}
        if(!bspatialX) {std::cout << "spatialX is not specified " << std::endl;}
        if(!bspatialY) {std::cout << "spatialY is not specified " << std::endl;}
        if(!bhalfThetaLimit) {std::cout << "halfThetaLimit is not specified " << std::endl;}
        if(!bemitter_size) {std::cout << "emitter_size is not specified " << std::endl;}
        if(!bsensor_size) {std::cout << "sensor_size is not specified " << std::endl;}
        if(!bmediumLx) {std::cout << "mediumLx is not specified " << std::endl;}
        if(!bmediumRx) {std::cout << "mediumRx is not specified " << std::endl;}
        if(!bdistribution) {std::cout << "distribution is not specified " << std::endl;}
        if(!bgOrKappa) {std::cout << "gOrKappa is not specified " << std::endl;}
        if(!bemitter_distance) {std::cout << "emitter_distance is not specified " << std::endl;}
        if(!bemitter_lens_aperture) {std::cout << "emitter_lens_aperture is not specified " << std::endl;}
        if(!bemitter_lens_focalLength) {std::cout << "emitter_lens_focalLength is not specified " << std::endl;}
        if(!bemitter_lens_active) {std::cout << "emitter_lens_active is not specified " << std::endl;}
        if(!bsensor_distance) {std::cout << "sensor_distance is not specified " << std::endl;}
        if(!bsensor_lens_aperture) {std::cout << "sensor_lens_aperture is not specified " << std::endl;}
        if(!bsensor_lens_focalLength) {std::cout << "sensor_lens_focalLength is not specified " << std::endl;}
        if(!bsensor_lens_active) {std::cout << "sensor_lens_active is not specified " << std::endl;}
#ifdef SPLINE_RIF
        if(!brifgridFile) {std::cout << "rifgridFile is not specified " << std::endl;}
#endif
        if(!bprintInputs) {std::cout << "printInputs is not specified " << std::endl;}
        if(!(bthreads && bprecision && bnumPhotons && boutFilePrefix && bsigmaT && balbedo && bgVal && 
#ifdef FUS_RIF
            bf_u && bspeed_u && bn_o && bn_scaling && bn_coeff && bradius && bcenter1 && bcenter2 && bactive1 && bactive2 && bphase1 && bphase2 && bchordlength && btheta_sources && btrans_z_sources && 
#else
            bf_u && bspeed_u && bn_o && bn_max && bn_clip && bphi_min && bphi_max && bmode && 
#endif
            bemitter_gap && bsensor_gap && ber_stepsize && bdirectTol && buseInitializationHack && brrWeight && bprojectorTexture && buseDirect && buseAngularSampling && buseBounceDecomposition && bmaxDepth && bmaxPathlength && bpathLengthMin && bpathLengthMax && bpathLengthBins && bspatialX && bspatialY && bhalfThetaLimit && bemitter_size && bsensor_size && bmediumLx && bmediumRx && bdistribution && bgOrKappa && bemitter_distance && bemitter_lens_aperture && bemitter_lens_focalLength && bemitter_lens_active && bsensor_distance && bsensor_lens_aperture && bsensor_lens_focalLength && bsensor_lens_active && bprintInputs)){
            std::cout << "crashing as one or more inputs is absent" << std::endl;
            exit (EXIT_FAILURE);
        }
    }

    if(printInputs){
        std::cout << "numPhotons = "<< numPhotons   << std::endl;
        std::cout << "outFilePrefix = " << outFilePrefix        << std::endl;
        std::cout << "sigmaT = "    << sigmaT       << std::endl;
        std::cout << "albedo = "    << albedo       << std::endl;
        std::cout << "gVal = "      << gVal         << std::endl;
#ifdef FUS_RIF
        std::cout << "f_u = " << f_u << std::endl;
        std::cout << "speed_u = " << speed_u << std::endl;
        std::cout << "n_o = " << n_o << std::endl;
        std::cout << "n_scaling = " << n_scaling << std::endl;
        std::cout << "n_coeff = " << n_coeff << std::endl;
        std::cout << "radius = " << radius << std::endl;
        std::cout << "center1 = (" << center1[0] << ", " << center1[1] << ", " << center1[2] << ")" << std::endl;
        std::cout << "center2 = (" << center2[0] << ", " << center2[1] << ", " << center2[2] << ")" << std::endl;
        std::cout << "active1 = " << active1 << std::endl;
        std::cout << "active2 = " << active2 << std::endl;
        std::cout << "phase1 = " << phase1 << std::endl;
        std::cout << "phase2 = " << phase2 << std::endl;
        std::cout << "chordlength = " << chordlength << std::endl;
        std::cout << "theta_sources = " << theta_sources << std::endl;
        std::cout << "trans_z_sources = " << trans_z_sources << std::endl;
#else
        std::cout << "f_u = " << f_u << std::endl;
        std::cout << "speed_u = " << speed_u << std::endl;
        std::cout << "n_o = " << n_o << std::endl;
        std::cout << "n_max = " << n_max << std::endl;
        std::cout << "n_clip = " << n_clip << std::endl;
        std::cout << "phi_min = " << phi_min << std::endl;
        std::cout << "phi_max = " << phi_max << std::endl;
        std::cout << "mode = "      << mode         << std::endl;
#endif
        std::cout << "emitter_gap = "   << emitter_gap          << std::endl;
        std::cout << "sensor_gap = "    << sensor_gap           << std::endl;
        std::cout << "projectorTexture = "<< projectorTexture << std::endl;
        std::cout << "useDirect = " << useDirect << std::endl;
        std::cout << "useAngularSampling= " << useAngularSampling << std::endl;
        std::cout << "useBounceDecomposition= " << useBounceDecomposition << std::endl;
        std::cout << "maxDepth = " << maxDepth << std::endl;
        std::cout << "maxPathlength = " << maxPathlength << std::endl;
        std::cout << "Total medium length = " << mediumR[0] - mediumL[0] << std::endl;
        std::cout << "pathLengthMin = " << pathLengthMin << std::endl;
        std::cout << "pathLengthMax = " << pathLengthMax << std::endl;
        std::cout << "pathLengthBins = " << pathLengthBins << std::endl;
        std::cout << "spatialX = " << spatialX << std::endl;
        std::cout << "spatialY = " << spatialY << std::endl;
        std::cout << "halfThetaLimit = " << halfThetaLimit << std::endl;
        std::cout << "emitter_size = " << emitter_size << std::endl;
        std::cout << "sensor_size = " << sensor_size << std::endl;
        std::cout << "distribution = " << distribution << std::endl;
        std::cout << "gOrKappa = " << gOrKappa << std::endl;
        std::cout << "emitter_distance = " << emitter_distance << std::endl;
        std::cout << "emitter_lens_aperture = " << emitter_lens_aperture << std::endl;
        std::cout << "emitter_lens_focalLength = " << emitter_lens_focalLength << std::endl;
        std::cout << "emitter_lens_active = " << emitter_lens_active << std::endl;
        std::cout << "sensor_distance = " << sensor_distance << std::endl;
        std::cout << "sensor_lens_aperture = " << sensor_lens_aperture << std::endl;
        std::cout << "sensor_lens_focalLength = " << sensor_lens_focalLength << std::endl;
        std::cout << "sensor_lens_active = " << sensor_lens_active << std::endl;
#ifdef SPLINE_RIF
        std::cout << "rifgridFile = " << rifgridFile << std::endl;
#endif
        std::cout << "printInputs = " << printInputs << std::endl;
    }

    // some sanity checks
    {
        if(emitter_distance < 0) {std::cout << "emitter_distance = " << emitter_distance << " should be strictly non-zero" << std::endl; exit (EXIT_FAILURE);}
        if(sensor_distance < 0) {std::cout << "sensor_distance = " << sensor_distance << " should be strictly non-zero" << std::endl; exit (EXIT_FAILURE);}
        if(emitter_lens_active && emitter_distance < 1e-4){std::cout << "lens_active and emitter_distance = " << emitter_distance << ". emitter_distance should be strictly positive (>1e-4) " << std::endl; exit (EXIT_FAILURE);}
        if(sensor_lens_active && sensor_distance < 1e-4){std::cout << "lens_active and sensor_distance = " << sensor_distance << ". sensor_distance should be strictly positive (>1e-4) " << std::endl; exit (EXIT_FAILURE);}
        if(bemitter_gap && (emitter_gap < 0 || emitter_gap > (mediumR[0] - mediumL[0]))){std::cout << "invalid gap between the emitter and the US:" << emitter_gap << std::endl; exit (EXIT_FAILURE);}
        if(bsensor_gap && (sensor_gap < 0 || sensor_gap > (mediumR[0] - mediumL[0]))){std::cout << "invalid gap between the sensor and the US:" << sensor_gap << std::endl; exit (EXIT_FAILURE);}
        if(bemitter_gap && bsensor_gap && ((sensor_gap + emitter_gap) > (mediumR[0] - mediumL[0]))){std::cout << "sum of sensor and emitter gaps is more than the medium size; sum of gaps is " << (sensor_gap + emitter_gap) << std::endl; exit (EXIT_FAILURE);}
#ifndef FUS_RIF
        if(phi_max < phi_min){std::cout << "phi_max must be greater than or equal to phi min" << std::endl; exit (EXIT_FAILURE);}
#endif
    }


#ifdef FUS_RIF
    theta_min= -asin(chordlength/(2*radius));
    theta_max=  asin(chordlength/(2*radius));
    trans_z_min = -chordlength/2;
    trans_z_max =  chordlength/2; 
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
    const tvec::Vec3f axis_ux( FPCONST(0.0), FPCONST(0.0), FPCONST(1.0));
    const tvec::Vec3f p_u(FPCONST(0.0),  FPCONST(0.0), FPCONST(0.0));

    const med::Medium medium(sigmaT, albedo, phase);

    scn::Scene<tvec::TVector3> scene(ior, mediumL, mediumR,
                        lightOrigin, lightDir, halfThetaLimit, projectorTexture, lightPlane, Li,
                        viewOrigin, viewDir, viewX, viewPlane, pathlengthRange, useBounceDecomposition,
                        distribution, gOrKappa,
                        emitter_lens_origin, emitter_lens_aperture, emitter_lens_focalLength, emitter_lens_active,
                        sensor_lens_origin, sensor_lens_aperture, sensor_lens_focalLength, sensor_lens_active,
#ifdef FUS_RIF
                        f_u, speed_u, n_o, n_scaling, n_coeff, radius, center1, center2, active1, active2, phase1, phase2, theta_min, theta_max, theta_sources, trans_z_min, trans_z_max, trans_z_sources, 
#else
                        f_u, speed_u, n_o, n_max, n_clip, phi_min, phi_max, mode, 
#endif
                        axis_uz, axis_ux, p_u, er_stepsize, directTol, rrWeight, precision, EgapEndLocX, SgapBeginLocX, useInitializationHack
#ifdef SPLINE_RIF
                        , rifgridFile
//                      , xmin, xmax, N
#endif
                        );

    photon::Renderer<tvec::TVector3> renderer(maxDepth, maxPathlength, useDirect, useAngularSampling, threads);

    image::SmallImage img(viewReso.x, viewReso.y, viewReso.z);
    renderer.renderImage(img, medium, scene, numPhotons);

    img.writePFM3D(outFilePrefix+".pfm3d");

    delete phase;

    return 0;
}
