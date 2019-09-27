/*
 * renderer_sample.cpp
 *
 *  Created on: Nov 20, 2015
 *      Author: igkiou
 */

#include "renderer.h"
#include <sstream>

int main() {

	/*
	 * Initialize scattering parameters.
	 */
	const Float sigmaT = FPCONST(1);
	const Float albedo = FPCONST(0.9);
	const Float gVal = FPCONST(0);
	pfunc::HenyeyGreenstein *phase = new pfunc::HenyeyGreenstein(gVal);

	/*
	 * Initialize scene parameters.
	 */
	const Float ior = FPCONST(1.333);
	const tvec::Vec3f mediumL(-FPCONST(1.5), -FPCONST(5.0), -FPCONST(5.0));
	const tvec::Vec3f mediumR( FPCONST(1.5),  FPCONST(5.0),  FPCONST(5.0));

	/*
	 * Initialize source parameters.
	 */
	const tvec::Vec3f lightOrigin(mediumR[0], FPCONST(0.0), FPCONST(0.0));
//	const Float rayAngle = -FPCONST(0.5236);
	const Float lightAngle = FPCONST(M_PI);
//	const Float lightAngle = -FPCONST(1);
	const tvec::Vec3f lightDir(std::cos(lightAngle), std::sin(lightAngle),
							FPCONST(0.0));
 //   std::cout << "lightDir:" << lightDir[0] << "," << lightDir[1] << "," << lightDir[2] << "\n";
	const tvec::Vec2f lightPlane(FPCONST(0.1), FPCONST(0.1));
	const Float Li = FPCONST(75000);

	/*
	 * Initialize camera parameters.
	 */
	const tvec::Vec3f viewOrigin(mediumL[0], FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f viewDir(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f viewX(FPCONST(0.0), -FPCONST(1.0), FPCONST(0.0));
//	const tvec::Vec3f viewY(FPCONST(0.0), FPCONST(0.0), -FPCONST(1.0));
	const tvec::Vec2f viewPlane(FPCONST(5.0), FPCONST(5.0));
	const tvec::Vec2f pathlengthRange(FPCONST(0), FPCONST(64));
	const tvec::Vec3i viewReso(128, 128, 128);

	/*
	 * Initialize rendering parameters.
	 */
	const int64 numPhotons = 10000L;
//	const int64 numPhotons = 10000L;
//	const int64 numPhotons = 5L;
//	const int64 numPhotons = 20000000L;
	const int maxDepth = -1;
//	const int maxDepth = 1;
	const Float maxPathlength = -1;
	const bool useDirect = false;

	/*
	 * Initialize US parameters
	 */
	const Float f_u = 848*1e3;
	const Float speed_u = 1500;

	const Float n_o = 1.3333;
	const Float n_max = 0;
//	const Float n_max = 0;
	const int mode = 0;

	const tvec::Vec3f axis_uz(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f axis_ux( FPCONST(0.0), FPCONST(0.0), FPCONST(1.0));
	const tvec::Vec3f p_u(FPCONST(0.0),  FPCONST(0.0), FPCONST(0.0));

	const Float er_stepsize = 1e-3;

//	printf("\nnum photons = %ld\n", numPhotons);
//	printf("max depth = %d\n", maxDepth);
//	printf("max pathlength= %lf\n", maxPathlength);

	const med::Medium medium(sigmaT, albedo, phase);

//	std::string lightTextureFile("/home/igkiou/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr/renderer/images/ABCD.pfm");
	std::string lightTextureFile("/home/igkiou/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr/renderer/images/White.pfm");

	const scn::Scene<tvec::TVector3> scene(ior, mediumL, mediumR,
						lightOrigin, lightDir, lightTextureFile, lightPlane, Li,
						viewOrigin, viewDir, viewX, viewPlane, pathlengthRange,
						f_u, speed_u, n_o, n_max, mode, axis_uz, axis_ux, p_u, er_stepsize);

	photon::Renderer<tvec::TVector3> renderer(maxDepth, maxPathlength, useDirect);


	image::SmallImage img(viewReso.x, viewReso.y, viewReso.z);
	renderer.renderImage(img, medium, scene, numPhotons);

	img.writeToFile("USOCTRendering");

	delete phase;

	return 0;
}
