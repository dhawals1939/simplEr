
addpath(genpath('../'));

%%%%%%%%%%%%%%%%%%
%%%% setup problem

% all units are in mm
% see 'sceneparams' and 'rendererparams' for descriptions of the various
% parameters

%% scattering medium

% simulated medium
sigmaT = .1;
albedo = 0.9;
gVal = 0;

%% basic scene info
% index of refraction
iorMedium = 1.3333;
% depth, width, and height of medium
mediumDimensions = [3 1000 1000];

%% lighting parameters
% frontLightFlag = 1 for frontlighting
frontLightFlag = 1;
lightAngle = deg2rad(180);
% lightFrontFlag = 1;
% center of source
lightOrigin = [0.0 0.0];
% lightAngle = deg2rad(225);
% size of source
% lightPlane = [mediumDimensions(2:3)];
lightPlane = [.1 .1];
Li = 75000.0;

%% camera parameters
% center of sensor
viewOrigin = [0.0 0.0];
viewAngle = 0;
% size of sensor
viewPlane = [5; 5];
% pathlengthRange = [-1; -1];
% viewReso = [128; 1; 1];
pathlengthRange = [2.95; 3.5];
% pathlengthRange = [0; 30];
viewReso = [128; 128; 128];

%% renderer options
% numPhotons = 100;
% numPhotons = 10000000;
numPhotons = 1e8;
% numPhotons = 11000000;
maxDepth = -1;
maxPathlength = -1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% do not edit below here
%% create scene params
viewOrigin = [-mediumDimensions(1) / 2; viewOrigin(:)];
viewDir = -[cos(viewAngle); sin(viewAngle); 0.0];
viewHorizontal = [0.0; -1.0; 0.0];
		
lightDir = [cos(lightAngle); sin(lightAngle); 0.0];
if (frontLightFlag == 0),
	lightOrigin = [-mediumDimensions(1) / 2; lightOrigin(:)];
else
	lightOrigin = [mediumDimensions(1) / 2; lightOrigin(:)];
end;
scene = scene3dparams('iorMedium', iorMedium, 'mediumDimensions', mediumDimensions,...
	'lightOrigin', lightOrigin, 'lightDir', lightDir, 'lightPlane', lightPlane, 'Li', Li,...
	'viewOrigin', viewOrigin, 'viewDir', viewDir, 'viewHorizontal', viewHorizontal,...
	'viewPlane', viewPlane, 'pathlengthRange', pathlengthRange, 'viewReso', viewReso);

%% create renderer params
useDirect = 0;			% Not implemented, always keep 0.
renderer = rendererparams('useDirect', useDirect', 'numPhotons', numPhotons,...
			'maxDepth', maxDepth, 'maxPathlength', maxPathlength);

%% do rendering
% im and im_alt should be numerically identical;
% im and im_altw should be the same up to noise, and identical if the
%		simulated and sampling mediums are the same;
% dX and dXw should be the same up to noise, and identical if the
%		simulated and sampling mediums are the same;

% render an image by importance sampling the simulated medium
imss = renderImage3d(sigmaT, albedo, gVal, scene, renderer);
