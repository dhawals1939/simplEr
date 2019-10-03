/*
 * renderer_sample.cpp
 *
 *  Created on: Nov 20, 2015
 *      Author: igkiou
 */

#include "renderer.h"
#include <sstream>

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

	/*
	 * Initialize rendering parameters.
	 */
	int64 numPhotons = 500L;
	bool useDirect = true;
	int maxDepth = -1;
	Float maxPathlength = -1;// This is for tracing. FIXME: inconsistent and redundant
	Float pathLengthMin = 0; // This is for path length binning
	Float pathLengthMax = 64; // This is for path length binning
	int pathLengthBins = 128;

	/*
	 * output file prefix
	 */
    std::string outFilePrefix = "USOCTRendering";

	/*
	 * Initialize scattering parameters.
	 */
	Float sigmaT = FPCONST(0);
	Float albedo = FPCONST(1.0);
	Float gVal = FPCONST(0);

	/*
	 * Initialize US parameters
	 */
	Float f_u = 867*1e3*2*M_PI;
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
		else{
			std::cerr << "Unknown variable in the input argument:" << param[0] << std::endl;
			std::cerr << "Should be on of "
					  << "numPhotons "
					  << "outFile "
					  << "sigmaT "
					  << "albedo "
					  << "gVal "
					  << "f_u "
					  << "speed_u "
					  << "n_o "
					  << "n_max "
					  << "mode "
					  << "projectorTexture "
					  << std::endl;
			return -1;
		}
	}

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
	std::cout << "maxDepth = " << maxDepth << std::endl;
	std::cout << "maxPathlength = " << maxPathlength << std::endl;
	std::cout << "pathLengthMin = " << pathLengthMin << std::endl;
	std::cout << "pathLengthMax = " << pathLengthMax << std::endl;
	std::cout << "pathLengthBins = " << pathLengthBins << std::endl;


	pfunc::HenyeyGreenstein *phase = new pfunc::HenyeyGreenstein(gVal);

	/*
	 * Initialize scene parameters.
	 */
	const Float ior = FPCONST(1.3333);
	const tvec::Vec3f mediumL(-FPCONST(1.5), -FPCONST(5.0), -FPCONST(5.0));
	const tvec::Vec3f mediumR( FPCONST(1.5),  FPCONST(5.0),  FPCONST(5.0));

	/*
	 * Initialize source parameters.
	 */
	const tvec::Vec3f lightOrigin(mediumR[0], FPCONST(0.0), FPCONST(0.0));
	const Float lightAngle = FPCONST(M_PI);
	const tvec::Vec3f lightDir(std::cos(lightAngle), std::sin(lightAngle),
							FPCONST(0.0));
	const tvec::Vec2f lightPlane(FPCONST(0.008), FPCONST(0.008));
	const Float Li = FPCONST(75000);

	/*
	 * Initialize camera parameters.
	 */
	const tvec::Vec3f viewOrigin(mediumL[0], FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f viewDir(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f viewX(FPCONST(0.0), -FPCONST(1.0), FPCONST(0.0));
	const tvec::Vec2f viewPlane(FPCONST(5.0), FPCONST(5.0));
	const tvec::Vec2f pathlengthRange(FPCONST(pathLengthMin), FPCONST(pathLengthMax));
	const tvec::Vec3i viewReso(128, 128, pathLengthBins);

	/*
	 * Initialize rendering parameters.
	 */
	const tvec::Vec3f axis_uz(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f axis_ux( FPCONST(0.0), FPCONST(0.0), FPCONST(1.0));
	const tvec::Vec3f p_u(FPCONST(0.0),  FPCONST(0.0), FPCONST(0.0));

	const Float er_stepsize = 1e-3;

	const med::Medium medium(sigmaT, albedo, phase);

	const scn::Scene<tvec::TVector3> scene(ior, mediumL, mediumR,
						lightOrigin, lightDir, projectorTexture, lightPlane, Li,
						viewOrigin, viewDir, viewX, viewPlane, pathlengthRange,
						f_u, speed_u, n_o, n_max, mode, axis_uz, axis_ux, p_u, er_stepsize);

	photon::Renderer<tvec::TVector3> renderer(maxDepth, maxPathlength, useDirect);

	image::SmallImage img(viewReso.x, viewReso.y, viewReso.z);
	renderer.renderImage(img, medium, scene, numPhotons);

	img.writeToFile(outFilePrefix);

	delete phase;

	return 0;
}
