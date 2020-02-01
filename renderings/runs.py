import os

# Backup
#globalcmd = "./samples/renderer3d_sample_bin threads=-1 \
#stricts=true \
#precision=4 \
#numPhotons=10000 \
#outFilePrefix=delete \
#sigmaT=0 \
#albedo=.9 \
#gVal=.9 \
#f_u=832000 \
#speed_u=1500 \
#n_o=1 \
#n_max=0 \
#mode=0 \
#er_stepsize=1e-3 \
#directTol=1e-6 \
#rrWeight=.001 \
#projectorTexture=images/CMU_36.pfm \
#useDirect=true \
#useAngularSampling=false \
#maxDepth=-1 \
#maxPathlength=100000 \
#pathLengthMin=0 \
#pathLengthMax=100000 \
#pathLengthBins=1 \
#spatialX=1024 \
#spatialY=1024 \
#mediumLx=-.15 \
#mediumRx=.15 \
#distribution=none \
#gOrKappa=1 \
#halfThetaLimit=1 \
#emitter_size=.3 \
#emitter_distance=.6 \
#emitter_lens_aperture=.3 \
#emitter_lens_focalLength=.3 \
#emitter_lens_active=true \
#sensor_size=.6 \
#sensor_distance=.3 \
#sensor_lens_aperture=.3 \
#sensor_lens_focalLength=.3 \
#sensor_lens_active=false \
#printInputs=true "

globalcmd = "samples/renderer3d_sample_bin threads=-1 \
stricts=true \
precision=4 \
numPhotons=10000 \
outFilePrefix=delete \
sigmaT=0 \
albedo=.9 \
gVal=.9 \
f_u=832000 \
speed_u=1500 \
n_o=1 \
n_max=0 \
mode=0 \
er_stepsize=1e-3 \
directTol=1e-6 \
rrWeight=.001 \
projectorTexture=../renderer/images/CMU_36.pfm \
useDirect=true \
useAngularSampling=false \
maxDepth=-1 \
maxPathlength=100000 \
pathLengthMin=0 \
pathLengthMax=100000 \
pathLengthBins=1 \
spatialX=1024 \
spatialY=1024 \
mediumLx=-.15 \
mediumRx=.15 \
distribution=none \
gOrKappa=1 \
halfThetaLimit=1 \
emitter_size=.3 \
emitter_distance=.6 \
emitter_lens_aperture=.3 \
emitter_lens_focalLength=.3 \
emitter_lens_active=true \
sensor_size=.6 \
sensor_distance=.3 \
sensor_lens_aperture=.3 \
sensor_lens_focalLength=.3 \
sensor_lens_active=false \
printInputs=true "

printCMDs = True 
executeCMDs = False 
AWS = False


#baseFolder = "/home/ubuntu/AOOCT_V1/"
baseFolder = "/home/igkiou/ercrdr_angletracingNEE/"
#baseFolder = "/mnt/da64b98f-9fd9-4b2c-994e-ca7276846901/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr_angletracingNEE/"

outFilePrefix= baseFolder + "renderings/MaysamRebuttal/ER"

runs = 10

sensor_distance=[".3", ".2", ".1"]


for r in range(0, runs):
    variablecmd = ""
    for s in range(len(sensor_distance)):
        variablecmd = "sensor_distance=" + sensor_distance[s] + " outFilePrefix=" + outFilePrefix + "_" + sensor_distance[s]
        cmd = baseFolder + "/renderer/" + globalcmd + variablecmd
        if printCMDs:
            print(cmd)
        if executeCMDs:
            if AWS:
                os.system("echo " + cmd + " > temp.sh")
                os.system("qsub temp.sh ")
            else:
                os.system(cmd)




#if (len(TransducerLengths) != len(n_maxs)):
#    print("length of transducer and n_maxs should match")
#    exit()
#
#for t in range(0, len(targets)):
#    for s in range(0, len(sigmas)):
#        for n in range(0,len(n_maxs)):
#            cmd = baseFolder + "renderer/samples/renderer3d_sample_bin projectorTexture=" + baseFolder + "renderer/images/White.pfm halfThetaLimit=0.0 gVal=0.9 albedo=0.99 numPhotons=" + str(numPhotons) + " useDirect=true lens_active=false mediumLx=-" + str(float(TransducerLengths[n])/2) + " mediumRx=" + str(float(TransducerLengths[n])/2) + " f_u=" + str(USOCTFreq) + " pathLengthMin=0 pathLengthMax=100 pathLengthBins=1 sigmaT=" + str(sigmas[s]) + " emitter_sensor_size=" + str(targets[t])  + " outFilePrefix=" + outFilePrefix + "_" + str(numPhotons) + "_" + str(USOCTFreq) + "_" + str(targets[t]) +  "_" + str(TransducerLengths[n]) + "_" + str(n_maxs[n]) + " printInputs=true"
#            if printCMDs:
#                print(cmd)
#            if executeCMDs:
#                if AWS:
#                    os.system("echo " + cmd + " > temp.sh")
#                    os.system("qsub temp.sh ")
#                else:
#                    os.system(cmd)
