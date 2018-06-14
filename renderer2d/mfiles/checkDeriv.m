%%%%%%%%%%%%%%%%%%
%%%% setup problem

% all units are in mm
% see 'sceneparams' and 'rendererparams' for descriptions of the various
% parameters

%% scattering medium

% simulated medium
sigmaT = 1;
albedo = 0.9;
gVal = 0.8;

% sampling medium (used for path sampling)
samplingSigmaT = sigmaT;
samplingAlbedo = albedo;
samplingGVal = gVal;
% samplingSigmaT = 1;
% samplingAlbedo = 0.95;
% samplingGVal = 0.5;

%% basic scene info
iorMedium = 1;
mediumDimensions = [2.5; 100; 100];

%% lighting directions

% frontLightFlagSet = {[0; 0; 0; 0], [0; 0; 1; 1], [1; 1; 1; 1]};
% lightAnglesSet = {[-5; -11.25; -22.5; -45], [-5; -22.5; 185; 202.5], [185; 191.25; 202.5; 225]};

% frontLightFlag = 1 for frontlighting
% frontLightFlag = 0;
% lightAngle = deg2rad(-45);
lightFrontFlag = 1;
lightAngle = deg2rad(225);
lightPlane = mediumDimensions(2:3);
Li = 75000.0;

%% viewing directions
% viewAngles = deg2rad([0; -10; -20]);
viewAngle = deg2rad(0);
viewOrigin = [0.0; 0.0];

%% renderer options
numPhotons = 10000000;
maxDepth = -1;
maxPathlength = -1;

%% image params

viewPlane = [50; 50];
pathlengthRange = [-1; -1];
viewReso = [128; 128; 1];
% pathlengthRange = [0; 100];
% viewReso = [128; 128; 128];

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% do not edit below here
%% create scene params
viewOrigin = [-mediumDimensions(1) / 2; viewOrigin(:)];
viewDir = -[cos(viewAngle); sin(viewAngle); 0];
viewX = [0.0; -1.0; 0.0];
viewY = [0.0; 0.0; -1.0];
		
lightDir = [cos(lightAngle); sin(lightAngle); 0];
if (frontLightFlag == 0),
	lightOrigin = [-mediumDimensions(1) / 2; 0.0; 0.0];
else
	lightOrigin = [mediumDimensions(1) / 2; 0.0; 0.0];
end;
scene = sceneparams('iorMedium', iorMedium, 'mediumDimensions', mediumDimensions,...
	'lightOrigin', lightOrigin, 'lightDir', lightDir, 'lightPlane', lightPlane, 'Li', Li,...
	'viewOrigin', viewOrigin, 'viewDir', viewDir, 'viewX', viewX, 'viewY', viewY,...
	'viewPlane', viewPlane, 'pathlengthRange', pathlengthRange, 'viewReso', viewReso);

%% create renderer params
useDirect = 0;			% Not implemented, always keep 0.
renderer = rendererparams('useDirect', useDirect', 'numPhotons', numPhotons,...
			'maxDepth', maxDepth, 'maxPathlength', maxPathlength);

%% check derivatives

materialParams = [sigmaT; albedo; gVal];
samplingMaterialParams = [samplingSigmaT; samplingAlbedo; samplingGVal];
e = 10^(-6);

[d dy dh] = checkgrad('obj_grad', materialParams, e, samplingMaterialParams,...
		scene, renderer)
