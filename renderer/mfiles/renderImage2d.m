function im = renderImage(sigmaT, albedo, gVal, scene, renderer)
%% 
% All units are in mm.

% sample
iorMedium = scene.iorMedium;
mediumDimensions = scene.mediumDimensions;

% light source
lightOrigin = scene.lightOrigin;
lightDir = scene.lightDir;
lightPlane = scene.lightPlane;
Li = scene.Li;

% camera and image
viewOrigin = scene.viewOrigin;
viewDir = scene.viewDir;
viewHorizontal = scene.viewHorizontal;
viewPlane = scene.viewPlane;
pathlengthRange = scene.pathlengthRange;
viewReso = scene.viewReso;

% renderer
numPhotons = renderer.numPhotons;
maxDepth = renderer.maxDepth;
maxPathlength = renderer.maxPathlength;
useDirect = renderer.useDirect;

%%
im = renderImage2d_mex(sigmaT, albedo, gVal, ...
		iorMedium, mediumDimensions, ...
		lightOrigin, lightDir, lightPlane, Li, ...
		viewOrigin, viewDir, viewHorizontal, viewPlane, pathlengthRange, viewReso, ...
		numPhotons, maxDepth, maxPathlength, useDirect);
im = permute(im, [2 1 3]);
