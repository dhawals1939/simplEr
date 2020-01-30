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
#include <sstream>

#include <vector>
#include "ceres/ceres.h"
#include "glog/logging.h"


using ceres::CostFunction;
using ceres::SizedCostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

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
	google::InitGoogleLogging(argv[0]);

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
	Float pathLengthMin = 0; // This is for path length binning
	Float pathLengthMax = 64; // This is for path length binning
	int pathLengthBins = 128;
	int spatialX = 128; // X resolution of the film 
	int spatialY = 128; // Y resolution of the film

	/*
	 * adhoc parameters -- Should be assigned to a better block. Very hacky now. 
	 */
	Float halfThetaLimit = FPCONST(12.8e-3);
	Float emitter_sensor_size = FPCONST(0.002); // sizes of sensor and emitter are equal and are square shaped

	/*
	 * Initialize scattering parameters.
	 */
	Float sigmaT = FPCONST(0);
	Float albedo = FPCONST(1.0);
	Float gVal = FPCONST(0);

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
	Float f_u = 848*1e3;
	Float speed_u = 1500;
	Float n_o = 1.3333;
	Float n_max = 1e-3;
	int mode = 0;
	Float er_stepsize = 1e-3;
	int precision = 8; // Number of dec. precision bits till which we accurately make er_step either because the sampled distances are not an integral multiple of the er_stepsize or because the boundary is hit before.
	Float directTol = 1e-5; // 10 um units
	Float rrWeight  = 1e-2; // only one in hundred survives second path call


	/*
	 * Spline approximation, spline parameters
	 */
#ifdef SPLINE_RIF
	Float xmin[] = {-0.01, -0.01};
	Float xmax[] = { 0.01,  0.01};
	int N[] = {21, 21};
#endif
	/*
	 * Projector texture
	 */
	std::string projectorTexture("/home/igkiou/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr/renderer/images/White.pfm");

	bool stricts=false;
	bool bthreads=false;
	bool bprecision=false;
	bool bnumPhotons=false;
	bool boutFilePrefix=false;
	bool bsigmaT=false;
	bool balbedo=false;
	bool bgVal=false;
	bool bf_u=false;
	bool bspeed_u=false;
	bool bn_o=false;
	bool bn_max=false;
	bool bmode=false;
	bool ber_stepsize=false;
	bool bdirectTol=false;
	bool brrWeight=false;
	bool bprojectorTexture=false;
	bool buseDirect=false;
	bool buseAngularSampling=false;
	bool bmaxDepth=false;
	bool bmaxPathlength=false;
	bool bpathLengthMin=false;
	bool bpathLengthMax=false;
	bool bpathLengthBins=false;
	bool bspatialX=false;
	bool bspatialY=false;
	bool bhalfThetaLimit=false;
	bool bemitter_sensor_size=false;
	bool bmediumLx=false;
	bool bmediumRx=false;
	bool bdistribution=false;
	bool bgOrKappa=false;
	bool bemitter_lens_aperture=false;
	bool bemitter_lens_focalLength=false;
	bool bemitter_lens_active=false;
	bool bsensor_lens_aperture=false;
	bool bsensor_lens_focalLength=false;
	bool bsensor_lens_active=false;
	bool bprintInputs=false;

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
		}else if(param[0].compare("mode")==0){
 			bmode=true;
			mode = stoi(param[1]);
		}else if(param[0].compare("er_stepsize")==0){
 			ber_stepsize=true;
			er_stepsize = stof(param[1]);
		}else if(param[0].compare("directTol")==0){
 			bdirectTol=true;
			directTol = stof(param[1]);
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
		}else if(param[0].compare("emitter_sensor_size")==0){
 			bemitter_sensor_size=true;
			emitter_sensor_size = stof(param[1]);
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
					  << "f_u, "
					  << "speed_u, "
					  << "n_o, "
					  << "n_max, "
					  << "mode, "
					  << "er_stepsize, "
					  << "directTol, "
					  << "rrWeight, "
					  << "projectorTexture, "
					  << "useDirect, "
					  << "useAngularSampling, "
					  << "maxDepth, "
					  << "maxPathlength, "
					  << "pathLengthMin, "
					  << "pathLengthMax, "
					  << "pathLengthBins, "
					  << "spatialX, "
					  << "spatialY, "
					  << "halfThetaLimit, "
					  << "emitter_sensor_size, "
					  << "mediumLx, "
					  << "mediumRx, "
					  << "distribution, "
					  << "gOrKappa, "
					  << "emitter_lens_aperture, "
					  << "emitter_lens_focalLength, "
					  << "emitter_lens_active, "
					  << "sensor_lens_aperture, "
					  << "sensor_lens_focalLength, "
					  << "sensor_lens_active, "
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
		if(!bf_u) {std::cout << "f_u is not specified " << std::endl;}
		if(!bspeed_u) {std::cout << "speed_u is not specified " << std::endl;}
		if(!bn_o) {std::cout << "n_o is not specified " << std::endl;}
		if(!bn_max) {std::cout << "n_max is not specified " << std::endl;}
		if(!bmode) {std::cout << "mode is not specified " << std::endl;}
		if(!ber_stepsize) {std::cout << "er_stepsize is not specified " << std::endl;}
		if(!bdirectTol) {std::cout << "directTol is not specified " << std::endl;}
		if(!brrWeight) {std::cout << "rrWeight is not specified " << std::endl;}
		if(!bprojectorTexture) {std::cout << "projectorTexture is not specified " << std::endl;}
		if(!buseDirect) {std::cout << "useDirect is not specified " << std::endl;}
		if(!buseAngularSampling) {std::cout << "useAngularSampling is not specified " << std::endl;}
		if(!bmaxDepth) {std::cout << "maxDepth is not specified " << std::endl;}
		if(!bmaxPathlength) {std::cout << "maxPathlength is not specified " << std::endl;}
		if(!bpathLengthMin) {std::cout << "pathLengthMin is not specified " << std::endl;}
		if(!bpathLengthMax) {std::cout << "pathLengthMax is not specified " << std::endl;}
		if(!bpathLengthBins) {std::cout << "pathLengthBins is not specified " << std::endl;}
		if(!bspatialX) {std::cout << "spatialX is not specified " << std::endl;}
		if(!bspatialY) {std::cout << "spatialY is not specified " << std::endl;}
		if(!bhalfThetaLimit) {std::cout << "halfThetaLimit is not specified " << std::endl;}
		if(!bemitter_sensor_size) {std::cout << "emitter_sensor_size is not specified " << std::endl;}
		if(!bmediumLx) {std::cout << "mediumLx is not specified " << std::endl;}
		if(!bmediumRx) {std::cout << "mediumRx is not specified " << std::endl;}
		if(!bdistribution) {std::cout << "distribution is not specified " << std::endl;}
		if(!bgOrKappa) {std::cout << "gOrKappa is not specified " << std::endl;}
		if(!bemitter_lens_aperture) {std::cout << "sensor_lens_aperture is not specified " << std::endl;}
		if(!bemitter_lens_focalLength) {std::cout << "sensor_lens_focalLength is not specified " << std::endl;}
		if(!bemitter_lens_active) {std::cout << "sensor_lens_active is not specified " << std::endl;}
		if(!bsensor_lens_aperture) {std::cout << "sensor_lens_aperture is not specified " << std::endl;}
		if(!bsensor_lens_focalLength) {std::cout << "sensor_lens_focalLength is not specified " << std::endl;}
		if(!bsensor_lens_active) {std::cout << "sensor_lens_active is not specified " << std::endl;}
		if(!bprintInputs) {std::cout << "printInputs is not specified " << std::endl;}

        if(!(bthreads && bprecision && bnumPhotons && boutFilePrefix && bsigmaT && balbedo && bgVal && bf_u && bspeed_u && bn_o && bn_max && bmode && ber_stepsize && bdirectTol && brrWeight && bprojectorTexture && buseDirect && buseAngularSampling && bmaxDepth && bmaxPathlength && bpathLengthMin && bpathLengthMax && bpathLengthBins && bspatialX && bspatialY && bhalfThetaLimit && bemitter_sensor_size && bmediumLx && bmediumRx && bdistribution && bgOrKappa && bemitter_lens_aperture && bemitter_lens_focalLength && bemitter_lens_active && bsensor_lens_aperture && bsensor_lens_focalLength && bsensor_lens_active && bprintInputs)){
            std::cout << "crashing as one or more inputs is absent" << std::endl;
            exit (EXIT_FAILURE);
        }
    }

	if(printInputs){
		std::cout << "numPhotons = "<< numPhotons 	<< std::endl;
		std::cout << "outFilePrefix = " << outFilePrefix 		<< std::endl;
		std::cout << "sigmaT = " 	<< sigmaT 		<< std::endl;
		std::cout << "albedo = " 	<< albedo 		<< std::endl;
		std::cout << "gVal = " 		<< gVal 		<< std::endl;
		std::cout << "f_u = " 		<< f_u 			<< std::endl;
		std::cout << "speed_u = " 	<< speed_u 		<< std::endl;
		std::cout << "n_o = " 		<< n_o 			<< std::endl;
		std::cout << "n_max  = " 	<< n_max  		<< std::endl;
		std::cout << "mode = " 		<< mode 		<< std::endl;
		std::cout << "projectorTexture = "<< projectorTexture << std::endl;
		std::cout << "useDirect = " << useDirect << std::endl;
		std::cout << "useAngularSampling= " << useAngularSampling << std::endl;
		std::cout << "maxDepth = " << maxDepth << std::endl;
		std::cout << "maxPathlength = " << maxPathlength << std::endl;
        std::cout << "Total medium length = " << mediumR[0] - mediumL[0] << std::endl;
		std::cout << "pathLengthMin = " << pathLengthMin << std::endl;
		std::cout << "pathLengthMax = " << pathLengthMax << std::endl;
		std::cout << "pathLengthBins = " << pathLengthBins << std::endl;
		std::cout << "spatialX = " << spatialX << std::endl;
		std::cout << "spatialY = " << spatialY << std::endl;
		std::cout << "halfThetaLimit = " << halfThetaLimit << std::endl;
		std::cout << "emitter_sensor_size = " << emitter_sensor_size << std::endl;
		std::cout << "distribution = " << distribution << std::endl;
		std::cout << "gOrKappa = " << gOrKappa << std::endl;
		std::cout << "emitter_lens_aperture = " << emitter_lens_aperture << std::endl;
		std::cout << "emitter_lens_focalLength = " << emitter_lens_focalLength << std::endl;
		std::cout << "emitter_lens_active = " << emitter_lens_active << std::endl;
		std::cout << "sensor_lens_aperture = " << sensor_lens_aperture << std::endl;
		std::cout << "sensor_lens_focalLength = " << sensor_lens_focalLength << std::endl;
		std::cout << "sensor_lens_active = " << sensor_lens_active << std::endl;
		std::cout << "printInputs = " << printInputs << std::endl;
    }
	pfunc::HenyeyGreenstein *phase = new pfunc::HenyeyGreenstein(gVal);

    tvec::Vec3f emitter_lens_origin(mediumR[0], FPCONST(0.0), FPCONST(0.0));
    tvec::Vec3f sensor_lens_origin(mediumL[0], FPCONST(0.0), FPCONST(0.0));

	/*
	 * Initialize source parameters.
	 */
	const tvec::Vec3f lightOrigin(mediumR[0], FPCONST(0.0), FPCONST(0.0));
	const Float lightAngle = FPCONST(M_PI);
	const tvec::Vec3f lightDir(std::cos(lightAngle), std::sin(lightAngle),
							FPCONST(0.0));
	const tvec::Vec2f lightPlane(FPCONST(emitter_sensor_size), FPCONST(emitter_sensor_size));
	const Float Li = FPCONST(75000);

	/*
	 * Initialize camera parameters.
	 */
	const tvec::Vec3f viewOrigin(mediumL[0], FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f viewDir(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f viewX(FPCONST(0.0), -FPCONST(1.0), FPCONST(0.0));
	const tvec::Vec2f viewPlane(FPCONST(emitter_sensor_size), FPCONST(emitter_sensor_size));
	const tvec::Vec2f pathlengthRange(FPCONST(pathLengthMin), FPCONST(pathLengthMax));
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
						viewOrigin, viewDir, viewX, viewPlane, pathlengthRange,
						distribution, gOrKappa,
						emitter_lens_origin, emitter_lens_aperture, emitter_lens_focalLength, emitter_lens_active,
						sensor_lens_origin, sensor_lens_aperture, sensor_lens_focalLength, sensor_lens_active,
						f_u, speed_u, n_o, n_max, mode, axis_uz, axis_ux, p_u, er_stepsize, directTol, rrWeight, precision
#ifdef SPLINE_RIF
						, xmin, xmax, N
#endif
						);


	photon::Renderer<tvec::TVector3> renderer(maxDepth, maxPathlength, useDirect, useAngularSampling, threads);

	image::SmallImage img(viewReso.x, viewReso.y, viewReso.z);
	renderer.renderImage(img, medium, scene, numPhotons);

	img.writePFM3D(outFilePrefix+".pfm3d");

	delete phase;

	return 0;
}
