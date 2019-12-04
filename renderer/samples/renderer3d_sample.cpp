/*
 * renderer_sample.cpp
 *
 *  Created on: Nov 20, 2015
 *      Author: igkiou
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

// A CostFunction implementing analytically derivatives for the
// function f(x) = 10 - x.
class QuadraticCostFunction
  : public SizedCostFunction<1 /* number of residuals */,
                             3 /* size of first parameter */> {
 public:
  QuadraticCostFunction(double k_[]) {k[0] = k_[0]; k[1] = k_[1]; k[2] = k_[2]; }
  virtual ~QuadraticCostFunction() {}

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    double x = parameters[0][0];
    double y = parameters[0][1];
    double z = parameters[0][2];

    // f(x) = 10 - x.
//    residuals[0] =  (x - k[0])*(x-k[0]) + 1 ;
//    residuals[0] =  1 ;
//    residuals[0] =  ( (x - k[0])*(x - k[0]) + (y - k[1])*(y - k[1])  )  + 1  ;
    residuals[0] =  ( (x - k[0])*(x - k[0]) + (y - k[1])*(y - k[1]) + (z - k[2])*(z - k[2]) )  + 1 ;

    // f'(x) = -1. Since there's only 1 parameter and that parameter
    // has 1 dimension, there is only 1 element to fill in the
    // jacobians.
    //
    // Since the Evaluate function can be called with the jacobians
    // pointer equal to NULL, the Evaluate function must check to see
    // if jacobians need to be computed.
    //
    // For this simple problem it is overkill to check if jacobians[0]
    // is NULL, but in general when writing more complex
    // CostFunctions, it is possible that Ceres may only demand the
    // derivatives w.r.t. a subset of the parameter blocks.
    if (jacobians != NULL && jacobians[0] != NULL) {
//      jacobians[0][0] = 0;
//      jacobians[0][1] = 0;
//      jacobians[0][2] = 0; 
      jacobians[0][0] =  2*(x - k[0]);
      jacobians[0][1] =  2*(y - k[1]);
      jacobians[0][2] =  2*(z - k[2]);
    }

    return true;
  }
 private: 
  double k[3];
};



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
//  google::InitGoogleLogging(argv[0]);
//
//  // The variable to solve for with its initial value. It will be
//  // mutated in place by the solver.
//  double k_[] = {-10, 20,  -3130.03};
//  double x[] = {0, 0, 0};
//  const double initial_x[3] = {x[0], x[1], x[2]};
//
//  // Build the problem.
//  Problem problem;
//
//  // Set up the only cost function (also known as residual).
//  CostFunction* cost_function = new QuadraticCostFunction(k_);
//  problem.AddResidualBlock(cost_function, NULL, x);
//
//  // Run the solver!
//  Solver::Options options;
//  options.check_gradients = false;
//  options.max_num_iterations = 1000;
//  options.minimizer_type = ceres::LINE_SEARCH;
//  options.line_search_direction_type = ceres::BFGS;
//  options.function_tolerance = 1e-18;
//  options.gradient_tolerance = 1e-3;
//  options.parameter_tolerance = 1e-10;
//  options.minimizer_progress_to_stdout = false;
//
///*  std::string Error;
//  options.IsValid(&Error);
//  std::cout << Error << std::endl;
//*/
//  Solver::Summary summary;
//  Solve(options, &problem, &summary);
//
////  std::cout << summary.BriefReport() << "\n";
//  std::cout << "x : " << initial_x[0] << ", " << initial_x[1] << ", " << initial_x[2] 
//            << " -> " << x[0] << ", " << x[1] << ", " << x[2] << "\n";
//
//  return 0;



//	/*
//	 * Initialize rendering parameters.
//	 */
//	int64 numPhotons = 500L;
//	bool useDirect = true;
//    bool useAngularSampling = true;
//	int maxDepth = -1;
//	Float maxPathlength = -1;// This is for tracing. FIXME: inconsistent and redundant
//	Float pathLengthMin = 0; // This is for path length binning
//	Float pathLengthMax = 64; // This is for path length binning
//	int pathLengthBins = 128;
//
//	int spatialX = 128; // X resolution of the sensor
//	int spatialY = 128; // Y resolution of the sensor
//
//	Float halfThetaLimit = FPCONST(12.8e-3);
//	Float emitter_sensor_size = FPCONST(0.002); // sizes of sensor and emitter are equal and are square shaped

	/*
	 * output file prefix
	 */
    std::string outFilePrefix = "USOCTRendering";

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
    Float lens_aperture = .015;
    Float lens_focalLength = .015;
    bool lens_active = false;

    bool printInputs = true;

	/*
	 * Initialize US parameters
	 */
	Float f_u = 848*1e3;
	Float speed_u = 1500;
	Float n_o = 1.3333;
	Float n_max = 1e-3;
	int mode = 0;

	/*
	 * Projector texture
	 */
	std::string projectorTexture("/home/igkiou/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr/renderer/images/White.pfm");

	for(int i = 1; i < argc; i++){
		std::vector<std::string> param = tokenize(argv[i], "=");
		if(param.size() != 2){
			std::cerr << "Input argument " << argv[i] << "should be in the format arg=value" << std::endl;
			return -1;
		}
		if(param[0].compare("numPhotons")==0)
			numPhotons = stoi(param[1]);
		else if(param[0].compare("outFilePrefix")==0)
			outFilePrefix = param[1];
		else if(param[0].compare("sigmaT")==0)
			sigmaT = stof(param[1]);
		else if(param[0].compare("albedo")==0)
			albedo = stof(param[1]);
		else if(param[0].compare("gVal")==0)
			gVal = stof(param[1]);
		else if(param[0].compare("f_u")==0)
			f_u = stof(param[1]);
		else if(param[0].compare("speed_u")==0)
			speed_u = stof(param[1]);
		else if(param[0].compare("n_o")==0)
			n_o = stof(param[1]);
		else if(param[0].compare("n_max")==0)
			n_max = stof(param[1]);
		else if(param[0].compare("mode")==0)
			mode = stoi(param[1]);
		else if(param[0].compare("projectorTexture")==0)
			projectorTexture = param[1];
		else if(param[0].compare("useDirect")==0){
			transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
			if(param[1].compare("true")==0)
				useDirect = true;
			else if(param[1].compare("false")==0)
				useDirect = false;
			else{
				std::cerr << "useDirect should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
				return -1;
			}
		}
		else if(param[0].compare("useAngularSampling")==0){
			transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
			if(param[1].compare("true")==0)
				useAngularSampling = true;
			else if(param[1].compare("false")==0)
				useAngularSampling = false;
			else{
				std::cerr << "useAngularSampling should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
				return -1;
			}
		}
		else if(param[0].compare("maxDepth")==0)
			maxDepth = stoi(param[1]);
		else if(param[0].compare("maxPathlength")==0)
			maxPathlength = stof(param[1]);
		else if(param[0].compare("pathLengthMin")==0)
			pathLengthMin = stof(param[1]);
		else if(param[0].compare("pathLengthMax")==0)
			pathLengthMax = stof(param[1]);
		else if(param[0].compare("pathLengthBins")==0)
			pathLengthBins = stoi(param[1]);
		else if(param[0].compare("spatialX")==0)
			spatialX = stoi(param[1]);
		else if(param[0].compare("spatialY")==0)
			spatialY = stoi(param[1]);
		else if(param[0].compare("halfThetaLimit")==0)
			halfThetaLimit = stof(param[1]);
		else if(param[0].compare("emitter_sensor_size")==0)
			emitter_sensor_size = stof(param[1]);
		else if(param[0].compare("mediumLx")==0)
			mediumL[0] = stof(param[1]);
		else if(param[0].compare("mediumRx")==0)
			mediumR[0] = stof(param[1]);
		else if(param[0].compare("distribution")==0)
			distribution = (param[1]);
		else if(param[0].compare("gOrKappa")==0)
			gOrKappa = stof(param[1]);
		else if(param[0].compare("lens_aperture")==0)
			lens_aperture = stof(param[1]);
		else if(param[0].compare("lens_focalLength")==0)
			lens_focalLength = stof(param[1]);
		else if(param[0].compare("printInputs")==0){
			transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
			if(param[1].compare("true")==0)
				printInputs = true;
			else if(param[1].compare("false")==0)
				printInputs = false;
			else{
				std::cerr << "printInputs should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
				return -1;
			}
		}
		else if(param[0].compare("lens_active")==0){
			transform(param[1].begin(), param[1].end(), param[1].begin(), ::tolower);
			if(param[1].compare("true")==0)
				lens_active = true;
			else if(param[1].compare("false")==0)
				lens_active = false;
			else{
				std::cerr << "lens_active should be either true or false; Argument " << param[1] << " not recognized" << std::endl;
				return -1;
			}
		}
		else{
			std::cerr << "Unknown variable in the input argument:" << param[0] << std::endl;
			std::cerr << "Should be one of "
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
					  << "distribution, "
					  << "gOrKappa, "
					  << "lens_aperture, "
					  << "lens_focalLength, "
					  << "lens_active, "
					  << "printInputs "
                      << std::endl;
            return -1;
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
		std::cout << "pathLengthMin = " << pathLengthMin << std::endl;
		std::cout << "pathLengthMax = " << pathLengthMax << std::endl;
		std::cout << "pathLengthBins = " << pathLengthBins << std::endl;
		std::cout << "spatialX = " << spatialX << std::endl;
		std::cout << "spatialY = " << spatialY << std::endl;
		std::cout << "halfThetaLimit = " << halfThetaLimit << std::endl;
		std::cout << "emitter_sensor_size = " << emitter_sensor_size << std::endl;
		std::cout << "distribution = " << distribution << std::endl;
		std::cout << "gOrKappa = " << gOrKappa << std::endl;
		std::cout << "lens_aperture = " << lens_aperture << std::endl;
		std::cout << "lens_focalLength = " << lens_focalLength << std::endl;
		std::cout << "lens_active = " << lens_active << std::endl;
		std::cout << "printInputs = " << printInputs << std::endl;
    }
	pfunc::HenyeyGreenstein *phase = new pfunc::HenyeyGreenstein(gVal);

    tvec::Vec3f lens_origin(mediumL[0], FPCONST(0.0), FPCONST(0.0));

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

	const Float er_stepsize = 1e-3;

	const med::Medium medium(sigmaT, albedo, phase);

	const scn::Scene<tvec::TVector3> scene(ior, mediumL, mediumR,
						lightOrigin, lightDir, halfThetaLimit, projectorTexture, lightPlane, Li,
						viewOrigin, viewDir, viewX, viewPlane, pathlengthRange,
						f_u, speed_u, n_o, n_max, mode, axis_uz, axis_ux, p_u, er_stepsize);

	photon::Renderer<tvec::TVector3> renderer(maxDepth, maxPathlength, useDirect, useAngularSampling);

	image::SmallImage img(viewReso.x, viewReso.y, viewReso.z);
	renderer.renderImage(img, medium, scene, numPhotons);

//	img.writeToFile(outFilePrefix);
	img.writePFM3D(outFilePrefix+".pfm3d");

	delete phase;

	return 0;
}
