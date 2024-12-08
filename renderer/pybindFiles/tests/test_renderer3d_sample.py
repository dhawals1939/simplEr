
import matplotlib.pyplot as plt
import numpy as np

import math
import sys
import os
this_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_dir + '/..')

import tvector_pybind, scene_pybind, photon_pybind, phase_pybind, medium_pybind, image_pybind

outFilePrefix = 'USOCTRendering'

# default film parameters
pathLengthMin = 0
pathLengthMax = 64
pathLengthBins = 128
spatialX = 128
spatialY = 128

# adhoc parameters
halfThetaLimit = 12.8e-3
emitter_size = 0.002
sensor_size = 0.002
emitter_distance = 0.0
sensor_distance = 0.0

# default scattering parameters
sigmaT = 0.0
albedo = 1.0
gVal = 0.0


# default scene parameters
ior = 1.333
mediumL = tvector_pybind.Vec3f(-0.015, -5.0, -5.0)
mediumR = tvector_pybind.Vec3f( 0.015,  5.0,  5.0)

# default rendering parameters
numPhotons = 10000
maxDepth = -1
maxPathLength = -1
useDirect = False
useAngularSampling = True

# default final path importance sampling parameters
distribution = "vmf"
gOrKappa = 4
emitter_lens_aperture = .015
emitter_lens_focalLength = .015
emitter_lens_active = False

sensor_lens_aperture = .015
sensor_lens_focalLength = .015
sensor_lens_active = False

printInputs = True

f_u = 5*1e6
speed_u = 1500
n_o = 1.3333
n_scaling = 0.05e-3
n_coeff = 1
radius = 2 * 25.4e-3
center1 = tvector_pybind.Vec3f(-radius, 0., 0)
center2 = tvector_pybind.Vec3f(-radius, 0., 0)
active1 = True
active2 = True
phase1 = 0
phase2 = 0
chordlength = 0.5 * 25.4e-3
theta_min = -math.asin(chordlength/(2*radius))
theta_max =  math.asin(chordlength/(2*radius))
theta_sources = 100
trans_z_min = -chordlength/2
trans_z_max =  chordlength/2
trans_z_sources = 501

emitter_gap = .0
sensor_gap = .0
er_stepsize = 1e-3
precision = 4
directTol = 1e-5
useInitializationHack = True
rrWeight = 1e-2

useBounceDecomposition = True
projectorTexture = "/home/apedired/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr/renderer/images/White.pfm"

threads = -1
printInputs = False 
# WIP to read defaults from a file
# read in custom parameters, if any
#print('Reading custom parameters...')
custom_params_f = open(this_dir + '/renderer3d_sample_input.txt')
for line in custom_params_f:
    #print('\t%s' %(line.strip()))
    param, value = line.strip().split('=')
    if value == '': continue
    if param == 'threads': threads = int(value)
    elif param == 'precision': precision = int(value)
    elif param == 'numPhotons': numPhotons = int(value)
    elif param == 'outFilePrefix': outFilePrefix = value
    elif param == 'sigmaT': sigmaT = float(value)
    elif param == 'albedo': albedo = float(value)
    elif param == 'gVal': gVal = float(value)
    elif param == 'f_u': f_u = float(value)
    elif param == 'speed_u': speed_u = float(value)
    elif param == 'n_o': n_o = float(value)
    elif param == 'n_scaling': n_scaling = float(value)
    elif param == 'n_coeff': n_coeff = int(value)
    elif param == 'radius': radius = float(value)
    elif param == 'center1': center1 = tvector_pybind.Vec3f(float(value.strip().split(',')[0]), float(value.strip().split(',')[1]), float(value.strip().split(',')[2]))
    elif param == 'center2': center2 = tvector_pybind.Vec3f(float(value.strip().split(',')[0]), float(value.strip().split(',')[1]), float(value.strip().split(',')[2]))
    elif param == 'active1': active1 = (value == 'True')
    elif param == 'active2': active2 = (value == 'True')
    elif param == 'phase1': phase1 = float(value)
    elif param == 'phase2': phase2 = float(value)
    elif param == 'chordlength': chordlength = float(value)
    elif param == 'theta_sources': theta_sources = int(value)
    elif param == 'trans_z_sources': trans_z_sources = int(value)
    elif param == 'er_stepsize': er_stepsize = float(value)
    elif param == 'directTol': directTol = float(value)
    elif param == 'rrWeight': rrWeight = float(value)
    elif param == 'projectorTexture': projectorTexture = value
    elif param == 'useDirect': useDirect = (value == 'True')
    elif param == 'useAngularSampling': useAngularSampling = (value == 'True')
    elif param == 'useBounceDecomposition': useBounceDecomposition = (value == 'True')
    elif param == 'maxDepth': maxDepth = int(value)
    elif param == 'maxPathlength': maxPathlength = float(value)
    elif param == 'pathLengthMin': pathLengthMin = float(value)
    elif param == 'pathLengthMax': pathLengthMax = float(value)
    elif param == 'pathLengthBins': pathLengthBins = int(value)
    elif param == 'spatialX': spatialX = int(value)
    elif param == 'spatialY': spatialY = int(value)
    elif param == 'mediumLx': mediumLx = float(value)
    elif param == 'mediumRx': mediumRx = float(value)
    elif param == 'distribution': distribution = value
    elif param == 'gOrKappa': gOrKappa = float(value)
    elif param == 'halfThetaLimit': halfThetaLimit = float(value)
    elif param == 'emitter_gap': emitter_gap = float(value)
    elif param == 'sensor_gap': sensor_gap = float(value)
    elif param == 'emitter_size': emitter_size = float(value)
    elif param == 'emitter_distance': emitter_distance = float(value)
    elif param == 'emitter_lens_aperture': emitter_lens_aperture = float(value)
    elif param == 'emitter_lens_focalLength': emitter_lens_focalLength = float(value)
    elif param == 'emitter_lens_active': emitter_lens_active = (value == 'True')
    elif param == 'sensor_size': sensor_size = float(value)
    elif param == 'sensor_distance': sensor_distance = float(value)
    elif param == 'sensor_lens_aperture': sensor_lens_aperture = float(value)
    elif param == 'sensor_lens_focalLength': sensor_lens_focalLength = float(value)
    elif param == 'sensor_lens_active': sensor_lens_active = (value == 'True')
    elif param == 'useInitializationHack': useInitializationHack = (value == 'True')
    elif param == 'printInputs': printInputs = value
    else: print('\tError: \'%s\' is not a valid parameter' %param)

if printInputs:
    print('threads:' + str(threads))
    print('precision:' + str(precision))
    print('numPhotons:' + str(numPhotons))
    print('outFilePrefix:' + str(outFilePrefix))
    print('sigmaT:' + str(sigmaT))
    print('albedo:' + str(albedo))
    print('gVal:' + str(gVal))
    print('f_u:' + str(f_u))
    print('speed_u:' + str(speed_u))
    print('n_o:' + str(n_o))
    print('n_scaling:' + str(n_scaling))
    print('n_coeff:' + str(n_coeff))
    print('radius:' + str(radius))
    print('center1:(' + str(center1.index(0)) + ',' + str(center1.index(1)) + ',' + str(center1.index(2)) + ')')
    print('center2:(' + str(center2.index(0)) + ',' + str(center2.index(1)) + ',' + str(center2.index(2)) + ')')
    print('active1:' + str(active1))
    print('active2:' + str(active2))
    print('phase1:' + str(phase1))
    print('phase2:' + str(phase2))
    print('chordlength:' + str(chordlength))
    print('theta_sources:' + str(theta_sources))
    print('trans_z_sources:' + str(trans_z_sources))
    print('er_stepsize:' + str(er_stepsize))
    print('directTol:' + str(directTol))
    print('rrWeight:' + str(rrWeight))
    print('projectorTexture:' + str(projectorTexture))
    print('useDirect:' + str(useDirect))
    print('useAngularSampling:' + str(useAngularSampling))
    print('useBounceDecomposition:' + str(useBounceDecomposition))
    print('maxDepth:' + str(maxDepth))
    print('maxPathlength:' + str(maxPathlength))
    print('pathLengthMin:' + str(pathLengthMin))
    print('pathLengthMax:' + str(pathLengthMax))
    print('pathLengthBins:' + str(pathLengthBins))
    print('spatialX:' + str(spatialX))
    print('spatialY:' + str(spatialY))
    print('mediumLx:' + str(mediumLx))
    print('mediumRx:' + str(mediumRx))
    print('distribution:' + str(distribution))
    print('gOrKappa:' + str(gOrKappa))
    print('halfThetaLimit:' + str(halfThetaLimit))
    print('emitter_gap:' + str(emitter_gap))
    print('sensor_gap:' + str(sensor_gap))
    print('emitter_size:' + str(emitter_size))
    print('emitter_distance:' + str(emitter_distance))
    print('emitter_lens_aperture:' + str(emitter_lens_aperture))
    print('emitter_lens_focalLength:' + str(emitter_lens_focalLength))
    print('emitter_lens_active:' + str(emitter_lens_active))
    print('sensor_size:' + str(sensor_size))
    print('sensor_distance:' + str(sensor_distance))
    print('sensor_lens_aperture:' + str(sensor_lens_aperture))
    print('sensor_lens_focalLength:' + str(sensor_lens_focalLength))
    print('sensor_lens_active:' + str(sensor_lens_active))
    print('useInitializationHack:' + str(useInitializationHack))

# initialize transducer parameters
theta_min= -math.asin(chordlength/(2*radius))
theta_max=  math.asin(chordlength/(2*radius))
trans_z_min = -chordlength/2
trans_z_max =  chordlength/2 

phase = phase_pybind.HenyeyGreenstein(gVal)

emitter_lens_origin = tvector_pybind.Vec3f(mediumR.index(0), 0.0, 0.0)
EgapEndLocX = emitter_lens_origin.index(0) - emitter_gap
sensor_lens_origin = tvector_pybind.Vec3f(mediumL.index(0), 0.0, 0.0)
SgapBeginLocX = sensor_lens_origin.index(0) + sensor_gap 

# initialize source parameters
lightOrigin = tvector_pybind.Vec3f(mediumR.index(0) + emitter_distance, 0.0, 0.0)
lightAngle = math.pi
lightDir = tvector_pybind.Vec3f(math.cos(lightAngle), math.sin(lightAngle), 0.0)
lightPlane = tvector_pybind.Vec2f(emitter_size, emitter_size)
Li = 75000.0

# initialize camera parameters
viewOrigin = tvector_pybind.Vec3f(mediumL.index(0)-sensor_distance, 0.0, 0.0)
viewDir = tvector_pybind.Vec3f(-1.0, 0.0, 0.0)
viewX = tvector_pybind.Vec3f(0.0, -1.0, 0.0)
viewPlane = tvector_pybind.Vec2f(emitter_size, emitter_size)
pathlengthRange = tvector_pybind.Vec2f(pathLengthMin, pathLengthMax)
     
viewReso = tvector_pybind.Vec3i(spatialX, spatialY, pathLengthBins)

# initialize rendering parameters.
axis_uz = tvector_pybind.Vec3f(-1.0, 0.0, 0.0) 
axis_ux = tvector_pybind.Vec3f(0.0, 0.0, 1.0) 
p_u = tvector_pybind.Vec3f(0.0, 0.0, 0.0) 

medium = medium_pybind.Medium(sigmaT, albedo, phase)

# set up everything
print('Setting up scene...')
scene = scene_pybind.Scene(
    ior,
    mediumL,
    mediumR,
    lightOrigin,
    lightDir,
    halfThetaLimit,
    projectorTexture,
    lightPlane,
    Li,
    viewOrigin,
    viewDir,
    viewX,
    viewPlane,
    pathlengthRange,
    useBounceDecomposition,
    distribution,
    gOrKappa,
    emitter_lens_origin,
    emitter_lens_aperture,
    emitter_lens_focalLength,
    emitter_lens_active,
    sensor_lens_origin,
    sensor_lens_aperture,
    sensor_lens_focalLength,
    sensor_lens_active,
    f_u,
    speed_u,
    n_o,
    n_scaling,
    n_coeff,
    radius,
    center1,
    center2,
    active1,
    active2,
    phase1,
    phase2,
    theta_min,
    theta_max,
    theta_sources,
    trans_z_min,
    trans_z_max,
    trans_z_sources,
    axis_uz,
    axis_ux,
    p_u,
    er_stepsize,
    directTol,
    rrWeight, 
    precision, 
    EgapEndLocX, 
    SgapBeginLocX, 
    useInitializationHack
)

renderer = photon_pybind.Renderer(
    maxDepth, maxPathLength, useDirect, useAngularSampling, threads
)

img = image_pybind.SmallImage(
    viewReso.x,
    viewReso.y,
    viewReso.z
)

print('Rendering image...')
#print(type(img))
#print(type(medium))
#print(type(scene))
#print(type(numPhotons))
renderer.renderImage(img, medium, scene, numPhotons)
print('Rendering completed.')

# Adi: very slow and painful way to conver image to numpy array
(xres, yres, zres) = (img.getXRes(), img.getYRes(), img.getZRes())
pixels = []
for x in range(xres):
    for y in range(yres):
        for z in range(zres):
            pixels.append(img.getPixel(x, y, z))

img_arr = np.array(pixels)
img_arr = img_arr.reshape((xres, yres, zres))

import matplotlib
from matplotlib import image

pyplot.imshow(img_arr)
pyplot.show()


#photon_locs = np.nonzero(img_arr)
#
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(
#    photon_locs[0], photon_locs[1], photon_locs[2], zdir='z', s=10,
#)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#
#ax.set(xlim=(0, spatialX), ylim=(0, spatialY), zlim=(0, pathLengthBins))
#print('Displaying image, close the image window to exit the program')


plt.show()
