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
	Float er_stepsize = 1e-3;
	Float directTol = 1e-5; // 10 um units
	Float rrWeight  = 1e-2; // only one in hundred survives second path call


	/*
	 * Spline approximation, spline parameters
	 */
#ifdef SPLINE_RIF
	Float xmin[] = {-0.01, -0.01};
	Float xmax[] = { 0.01,  0.01};
	int N[] = {101, 101};
#endif
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
		else if(param[0].compare("er_stepsize")==0)
			er_stepsize = stof(param[1]);
		else if(param[0].compare("directTol")==0)
			directTol = stof(param[1]);
		else if(param[0].compare("rrWeight")==0)
			rrWeight = stof(param[1]);
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

	const med::Medium medium(sigmaT, albedo, phase);

	scn::Scene<tvec::TVector3> scene(ior, mediumL, mediumR,
						lightOrigin, lightDir, halfThetaLimit, projectorTexture, lightPlane, Li,
						viewOrigin, viewDir, viewX, viewPlane, pathlengthRange,
						f_u, speed_u, n_o, n_max, mode, axis_uz, axis_ux, p_u, er_stepsize, directTol, rrWeight
#ifdef SPLINE_RIF
						, xmin, xmax, N
#endif
						);

    tvec::Vec3f p;

    tvec::Vec3f db;
    tvec::Vec3f ds;

    Float sum = 0; 
    tvec::Vec3f vecsum(0.0); 

    Matrix3x3 matsum(0.0);

    for(Float pi = -0.001;pi <= 0.00101; pi += .0001){
        for(Float pj = -0.001;pj <= 0.00101; pj += .0001){
            for(Float pk = -0.001;pk <= 0.00101; pk += .0001){
//                std::cout << "( " << pi << ", " << pj << ", " << pk << "):";
                p.x = pi; p.y = pj; p.z = pk;
//                std::cout << scene.m_us.spline_RIF(p, 1) << ", " << scene.m_us.bessel_RIF(p, 1) << std::endl;
                sum += std::abs((scene.m_us.spline_RIF(p, 1) - scene.m_us.bessel_RIF(p, 1))/(scene.m_us.bessel_RIF(p, 1) + 1e-3));
                
                std::cout << scene.m_us.spline_dRIF(p, 1) << ", " << scene.m_us.bessel_dRIF(p, 1) << std::endl;               
                for(int i=0; i<3; i++){
                    vecsum[i] += std::abs( (scene.m_us.spline_dRIF(p, 1)[i] - scene.m_us.bessel_dRIF(p, 1)[i])/(std::abs(scene.m_us.bessel_dRIF(p, 1)[i]) + 1e-3)); 
                }
//               vecsum.x += std::abs(scene.m_us.spline_dRIF(p, 1).x - scene.m_us.bessel_dRIF(p, 1).x); 
//                vecsum.y += std::abs(scene.m_us.spline_dRIF(p, 1).y - scene.m_us.bessel_dRIF(p, 1).y); 
//                vecsum.z += std::abs(scene.m_us.spline_dRIF(p, 1).z - scene.m_us.bessel_dRIF(p, 1).z); 
                
//                std::cout << scene.m_us.spline_HessianRIF(p, 1).toString() << ", " << scene.m_us.bessel_HessianRIF(p, 1).toString() << std::endl; 
                for(int i=0; i<3; i++)
                    for(int j=0; j<3; j++)
                        matsum(i,j) = abs( (scene.m_us.spline_HessianRIF(p, 1)(i,j) - scene.m_us.bessel_HessianRIF(p, 1)(i,j))/(scene.m_us.bessel_HessianRIF(p, 1)(i,j) + 1e-3) );
            }
        }
    }

    std::cout << "sum:" << sum << std::endl;
    std::cout << "vecsum:" << vecsum << std::endl;
    std::cout << "matsum:" << matsum.toString() << std::endl;

	photon::Renderer<tvec::TVector3> renderer(maxDepth, maxPathlength, useDirect, useAngularSampling);

	image::SmallImage img(viewReso.x, viewReso.y, viewReso.z);
	renderer.renderImage(img, medium, scene, numPhotons);

	img.writePFM3D(outFilePrefix+".pfm3d");

	delete phase;

	return 0;
}
