/*
 * renderer_sample.cpp
 *
 *  Created on: Nov 20, 2015
 *      Author: igkiou
 */

#include "renderer.h"

int main() {

	/*
	 * Initialize scattering parameters.
	 */
	const Float sigmaT = FPCONST(10.0);
	const Float albedo = FPCONST(0.9);
	const Float gVal = FPCONST(0.8);
	pfunc::HenyeyGreenstein *phase = new pfunc::HenyeyGreenstein(gVal);

	/*
	 * Initialize sampling scattering parameters.
	 */
	const Float samplingSigmaT = FPCONST(1.0);
	const Float samplingAlbedo = FPCONST(0.95);
	const Float samplingGVal = FPCONST(0.0);
	pfunc::HenyeyGreenstein *samplingPhase = new pfunc::HenyeyGreenstein(
																samplingGVal);

	/*
	 * Initialize scene parameters.
	 */
	const Float ior = FPCONST(1.0);
	const tvec::Vec3f mediumL(-FPCONST(1.25), -FPCONST(10.0), -FPCONST(10.0));
	const tvec::Vec3f mediumR(FPCONST(1.25), FPCONST(10.0), FPCONST(10.0));

	/*
	 * Initialize source parameters.
	 */
	const tvec::Vec3f lightOrigin(mediumL.x, FPCONST(0.0), FPCONST(0.0));
//	const Float rayAngle = -FPCONST(0.5236);
	const Float lightAngle = FPCONST(0.0);
	const tvec::Vec3f lightDir(std::cos(lightAngle), std::sin(lightAngle),
							FPCONST(0.0));
	const tvec::Vec2f lightPlane(FPCONST(20.0), FPCONST(20.0));
	const Float Li = FPCONST(75000.0);

	/*
	 * Initialize camera parameters.
	 */
	const tvec::Vec3f viewOrigin(mediumR.x, FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f viewDir(-FPCONST(1.0), FPCONST(0.0), FPCONST(0.0));
	const tvec::Vec3f viewX(FPCONST(0.0), -FPCONST(1.0), FPCONST(0.0));
	const tvec::Vec3f viewY(FPCONST(0.0), FPCONST(0.0), -FPCONST(1.0));
	const tvec::Vec2f viewPlane(FPCONST(20.0), FPCONST(20.0));
	const tvec::Vec2f pathlengthRange(-FPCONST(1.0), -FPCONST(1.0));
	const tvec::Vec3i viewReso(128, 128, 1);

	/*
	 * Initialize rendering parameters.
	 */
	const int64 numPhotons = 5000000L;
//	const int64 numPhotons = 20000000L;
	const int maxDepth = -1;
	const Float maxPathlength = -1;
	const bool useDirect = false;

	printf("\nnum photons = %ld\n", numPhotons);
	printf("max depth = %d\n", maxDepth);
	printf("max pathlength= %lf\n", maxPathlength);

	const med::Medium medium(sigmaT, albedo, phase);
	const med::Medium samplingMedium(samplingSigmaT, samplingAlbedo,
									samplingPhase);
	const scn::Scene scene(ior, mediumL, mediumR,
						lightOrigin, lightDir, lightPlane, Li,
						viewOrigin, viewDir, viewX, viewY, viewPlane, pathlengthRange);

	photon::Renderer renderer(maxDepth, maxPathlength, useDirect);


	image::SmallImage img(viewReso.x, viewReso.y, viewReso.z);
	renderer.renderImage(img, medium, scene, numPhotons);

	image::SmallImage img_alt(viewReso.x, viewReso.y, viewReso.z);
	image::SmallImage dSigmaT(viewReso.x, viewReso.y, viewReso.z);
	image::SmallImage dAlbedo(viewReso.x, viewReso.y, viewReso.z);
	image::SmallImage dGVal(viewReso.x, viewReso.y, viewReso.z);
	renderer.renderDerivImage(img_alt, dSigmaT, dAlbedo, dGVal,
							medium, scene, numPhotons);

	image::SmallImage img_alt_weight(viewReso.x, viewReso.y, viewReso.z);
	image::SmallImage dSigmaT_weight(viewReso.x, viewReso.y, viewReso.z);
	image::SmallImage dAlbedo_weight(viewReso.x, viewReso.y, viewReso.z);
	image::SmallImage dGVal_weight(viewReso.x, viewReso.y, viewReso.z);
	renderer.renderDerivImageWeight(img_alt_weight, dSigmaT_weight,
								dAlbedo_weight, dGVal_weight,
								medium, samplingMedium, scene, numPhotons);

//	const std::string ofname("out.pfm");
//	img0.writeToFile(ofname);

	delete phase;

	return 0;
}
