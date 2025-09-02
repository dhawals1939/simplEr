/*
 * renderImage_mex.cpp
 *
 *  Created on: Dec 5, 2015
 *      Author: igkiou
 */

#include "renderer.h"
#include "mex_wrapper.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 19) {
		mexErrMsgTxt("Nineteen input arguments are required.");
	}

	/* Check number of output arguments */
	if (nlhs > 1) {
		mexErrMsgTxt("Too many output arguments.");
	}

	/* Input scattering parameters. */
	const double sigmaTd = (double) mxGetScalar(prhs[0]);
	const double albedod = (double) mxGetScalar(prhs[1]);
	const double gVald = (double) mxGetScalar(prhs[2]);

	/* Input medium parameters. */
	const double iord = (double) mxGetScalar(prhs[3]);
	const double *mediumDimensionsd = (double *) mxGetPr(prhs[4]);

	/* Input source parameters. */
	const double *lightOrigind = (double *) mxGetPr(prhs[5]);
	const double *lightDird = (double *) mxGetPr(prhs[6]);
	const double *lightPlaned = (double *) mxGetPr(prhs[7]);
	const double Lid = (double) mxGetScalar(prhs[8]);

	/* Input camera parameters. */
	const double *viewOrigind = (double *) mxGetPr(prhs[9]);
	const double *viewDird = (double *) mxGetPr(prhs[10]);
	const double *viewHorizontald = (double *) mxGetPr(prhs[11]);
	const double *viewPlaned = (double *) mxGetPr(prhs[12]);
	const double *pathlengthRanged = (double *) mxGetPr(prhs[13]);
	const double *viewResod = (double *) mxGetPr(prhs[14]);

	/* Input rendering parameters. */
	const double numPhotonsd = (double) mxGetScalar(prhs[15]);
	const double maxDepthd = (double) mxGetScalar(prhs[16]);
	const double maxPathlengthd = (double) mxGetScalar(prhs[17]);
	const double useDirectd = (double) mxGetScalar(prhs[18]);

	/*
	 * Initialize scattering parameters.
	 */
	pfunc::HenyeyGreenstein *phase = new pfunc::HenyeyGreenstein((Float) gVald);
	const Float sigmaT = (Float) sigmaTd;
	const Float albedo = (Float) albedod;

	/*
	 * Initialize scene parameters.
	 */
	const Float ior = (Float) iord;
	const tvec::Vec2f mediumL(- (Float) mediumDimensionsd[0] * FPCONST(0.5),
							- (Float) mediumDimensionsd[1] * FPCONST(0.5));
	const tvec::Vec2f mediumR((Float) mediumDimensionsd[0] * FPCONST(0.5),
							(Float) mediumDimensionsd[1] * FPCONST(0.5));

	/*
	 * Initialize source parameters.
	 */
	const tvec::Vec2f lightOrigin((Float) lightOrigind[0], (Float) lightOrigind[1]);
	const tvec::Vec2f lightDir((Float) lightDird[0], (Float) lightDird[1]);
	const tvec::Vec2f lightPlane((Float) lightPlaned[0], (Float) lightPlaned[1]);
	const Float Li = (Float) Lid;

	/*
	 * Initialize camera parameters.
	 */
	const tvec::Vec2f viewOrigin((Float) viewOrigind[0], (Float) viewOrigind[1]);
	const tvec::Vec2f viewDir((Float) viewDird[0], (Float) viewDird[1]);
	const tvec::Vec2f viewHorizontal((Float) viewHorizontald[0], (Float) viewHorizontald[1]);
	const tvec::Vec2f viewPlane((Float) viewPlaned[0], (Float) viewPlaned[1]);
	const tvec::Vec2f pathlengthRange((Float) pathlengthRanged[0], (Float) pathlengthRanged[1]);
	const tvec::Vec3i viewReso((int) viewResod[0], (int) viewResod[1], (int) viewResod[2]);

	/*
	 * Initialize rendering parameters.
	 */
	const int64 numPhotons = (int64) numPhotonsd;
	const int maxDepth = (int) maxDepthd;
	const Float maxPathlength = (Float) maxPathlengthd;
	const bool useDirect = (useDirectd > 0);

	const med::Medium medium(sigmaT, albedo, phase);
	const scn::Scene<tvec::TVector2> scene(ior, mediumL, mediumR,
						lightOrigin, lightDir, lightPlane, Li,
						viewOrigin, viewDir, viewHorizontal, viewPlane, pathlengthRange);
	image::SmallImage img0(viewReso.x, viewReso.y, viewReso.z);

	photon::Renderer<tvec::TVector2> renderer(maxDepth, maxPathlength, useDirect);
	renderer.renderImage(img0, medium, scene, numPhotons);

	/* Be sure to check for x and y here. */
#if USE_DOUBLE_PRECISION
	mxClassID matClassID = mxDOUBLE_CLASS;
#else
	mxClassID matClassID = mxSINGLE_CLASS;
#endif
	if (viewReso.z == 1) {
		plhs[0] = mxCreateNumericMatrix(viewReso.y, viewReso.x, matClassID, mxREAL);
	} else {
		mwSize dims[3];
		dims[0] = viewReso.y;
		dims[1] = viewReso.x;
		dims[2] = viewReso.z;
		plhs[0] = mxCreateNumericArray(static_cast<mwSize>(3), dims, matClassID, mxREAL);
	}
	Float *matImg = (Float *) mxGetData(plhs[0]);
	img0.copyImage(matImg, viewReso.y * viewReso.x * viewReso.z);
	delete phase;
}
