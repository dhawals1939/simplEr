/*
 * renderDerivImage_mex.cpp
 *
 *  Created on: Dec 5, 2015
 *      Author: igkiou
 */

#include "renderer.h"
#include "mex_wrapper.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 20) {
		mexErrMsgTxt("Twenty input arguments are required.");
	}

	/* Check number of output arguments */
	if (nlhs > 4) {
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
	const double *viewXd = (double *) mxGetPr(prhs[11]);
	const double *viewYd = (double *) mxGetPr(prhs[12]);
	const double *viewPlaned = (double *) mxGetPr(prhs[13]);
	const double *pathlengthRanged = (double *) mxGetPr(prhs[14]);
	const double *viewResod = (double *) mxGetPr(prhs[15]);

	/* Input rendering parameters. */
	const double numPhotonsd = (double) mxGetScalar(prhs[16]);
	const double maxDepthd = (double) mxGetScalar(prhs[17]);
	const double maxPathlengthd = (double) mxGetScalar(prhs[18]);
	const double useDirectd = (double) mxGetScalar(prhs[19]);

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
	const tvec::Vec3f mediumL(- (Float) mediumDimensionsd[0] * FPCONST(0.5),
							- (Float) mediumDimensionsd[1] * FPCONST(0.5),
							- (Float) mediumDimensionsd[2] * FPCONST(0.5));
	const tvec::Vec3f mediumR((Float) mediumDimensionsd[0] * FPCONST(0.5),
							(Float) mediumDimensionsd[1] * FPCONST(0.5),
							(Float) mediumDimensionsd[2] * FPCONST(0.5));

	/*
	 * Initialize source parameters.
	 */
	const tvec::Vec3f lightOrigin((Float) lightOrigind[0], (Float) lightOrigind[1], (Float) lightOrigind[2]);
	const tvec::Vec3f lightDir((Float) lightDird[0], (Float) lightDird[1], (Float) lightDird[2]);
	const tvec::Vec2f lightPlane((Float) lightPlaned[0], (Float) lightPlaned[1]);
	const Float Li = (Float) Lid;

	/*
	 * Initialize camera parameters.
	 */
	const tvec::Vec3f viewOrigin((Float) viewOrigind[0], (Float) viewOrigind[1], (Float) viewOrigind[2]);
	const tvec::Vec3f viewDir((Float) viewDird[0], (Float) viewDird[1], (Float) viewDird[2]);
	const tvec::Vec3f viewX((Float) viewXd[0], (Float) viewXd[1], (Float) viewXd[2]);
	const tvec::Vec3f viewY((Float) viewYd[0], (Float) viewYd[1], (Float) viewYd[2]);
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
	const scn::Scene scene(ior, mediumL, mediumR,
						lightOrigin, lightDir, lightPlane, Li,
						viewOrigin, viewDir, viewX, viewY, viewPlane, pathlengthRange);
	image::SmallImage img0(viewReso.x, viewReso.y, viewReso.z);
	image::SmallImage dSigmaT0(viewReso.x, viewReso.y, viewReso.z);
	image::SmallImage dAlbedo0(viewReso.x, viewReso.y, viewReso.z);
	image::SmallImage dGVal0(viewReso.x, viewReso.y, viewReso.z);

	photon::Renderer renderer(maxDepth, maxPathlength, useDirect);
	renderer.renderDerivImage(img0, dSigmaT0, dAlbedo0, dGVal0, medium, scene, numPhotons);

	/* Be sure to check for x and y here. */
#ifdef USE_DOUBLE_PRECISION
	mxClassID matClassID = mxDOUBLE_CLASS;
#else
	mxClassID matClassID = mxSINGLE_CLASS;
#endif
	if (viewReso.z == 1) {
		plhs[0] = mxCreateNumericMatrix(viewReso.y, viewReso.x, matClassID, mxREAL);
		plhs[1] = mxCreateNumericMatrix(viewReso.y, viewReso.x, matClassID, mxREAL);
		plhs[2] = mxCreateNumericMatrix(viewReso.y, viewReso.x, matClassID, mxREAL);
		plhs[3] = mxCreateNumericMatrix(viewReso.y, viewReso.x, matClassID, mxREAL);
	} else {
		mwSize dims[3];
		dims[0] = viewReso.y;
		dims[1] = viewReso.x;
		dims[2] = viewReso.z;
		plhs[0] = mxCreateNumericArray(static_cast<mwSize>(3), dims, matClassID, mxREAL);
		plhs[1] = mxCreateNumericArray(static_cast<mwSize>(3), dims, matClassID, mxREAL);
		plhs[2] = mxCreateNumericArray(static_cast<mwSize>(3), dims, matClassID, mxREAL);
		plhs[3] = mxCreateNumericArray(static_cast<mwSize>(3), dims, matClassID, mxREAL);
	}
	Float *matImg = (Float *) mxGetData(plhs[0]);
	img0.copyImage(matImg, viewReso.y * viewReso.x * viewReso.z);
	Float *matDSigmaT = (Float *) mxGetData(plhs[1]);
	dSigmaT0.copyImage(matDSigmaT, viewReso.y * viewReso.x * viewReso.z);
	Float *matDAlbedo = (Float *) mxGetData(plhs[2]);
	dAlbedo0.copyImage(matDAlbedo, viewReso.y * viewReso.x * viewReso.z);
	Float *matDGVal = (Float *) mxGetData(plhs[3]);
	dGVal0.copyImage(matDGVal, viewReso.y * viewReso.x * viewReso.z);
	delete phase;
}
