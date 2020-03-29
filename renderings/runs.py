import os

baseFolder = "/home/ubuntu/AOOCT_V1/"
#baseFolder = "/home/igkiou/ercrdr_angletracingNEE/"
#baseFolder = "/mnt/da64b98f-9fd9-4b2c-994e-ca7276846901/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr_angletracingNEE/"
globalcmd = baseFolder + "renderer/samples/renderer3d_sample_bin threads=-1 \
stricts=true \
threads=-1 \
precision=4 \
numPhotons=10000000 \
outFilePrefix=" + baseFolder + "delete \
sigmaT=166.6667 \
albedo=.99 \
gVal=.9 \
f_u=832000 \
speed_u=1500 \
n_o=1.3333 \
n_max=0.000657 \
phi_min=1.5708 \
phi_max=1.5708 \
mode=0 \
er_stepsize=1e-3 \
directTol=1e-6 \
rrWeight=.001 \
projectorTexture=" + baseFolder + "renderer/images/Disk_501.pfm \
useDirect=true \
useAngularSampling=True \
maxDepth=-1 \
maxPathlength=100000 \
pathLengthMin=0 \
pathLengthMax=100000 \
pathLengthBins=1 \
spatialX=1024 \
spatialY=1024 \
mediumLx=-.015 \
mediumRx=.015 \
distribution=none \
gOrKappa=1 \
halfThetaLimit=0 \
emitter_size=.0005 \
emitter_distance=0 \
emitter_lens_aperture=.3 \
emitter_lens_focalLength=.3 \
emitter_lens_active=false \
sensor_size=.005 \
sensor_distance=0 \
sensor_lens_aperture=.3 \
sensor_lens_focalLength=.3 \
sensor_lens_active=false \
gap=0 \
printInputs=true "
#projectorTexture=/home/ubuntu/AOOCT_V1/renderer/images/Disk_501.pfm \

printCMDs = True 
executeCMDs = False 
AWS = True 



outFilePrefix= baseFolder + "renderings/characterization/1M/ER_"

runs = 1
startIndex = 0

## 1. Transducer lengths
transducerlengths = ["0.005", "0.010", "0.015", "0.020", "0.025", "0.030", "0.035", "0.040", "0.045"] 
n_maxs = ["0.02333", "0.00579", "0.002535", "0.001447", ".0009253", "0.000657", ".000467676", ".000354545", ".000283838"]
if (len(transducerlengths) != len(n_maxs)):
    print("length of transducer and n_maxs should match")
    exit()

for r in range(startIndex, startIndex + runs):
    variablecmd = ""
    for s in range(len(transducerlengths)):
        variablecmd = " mediumLx=-" + str(float(transducerlengths[s])/2) + \
                      " mediumRx=" + str(float(transducerlengths[s])/2) + \
                      " n_max=" + n_maxs[s] + \
                      " sigmaT=" + str(5/float(transducerlengths[s])) + \
                      " outFilePrefix=" + outFilePrefix + "transducerlengths_" + transducerlengths[s] 
        cmd = globalcmd + variablecmd
        if printCMDs:
            print(cmd)
        if executeCMDs:
            if AWS:
                os.system("echo " + cmd + " > temp.sh")
                os.system("qsub temp.sh ")
            else:
                os.system(cmd)

del transducerlengths
del n_maxs
#
## 2. MFPs
#MFPs = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];
#
#for r in range(startIndex, startIndex + runs):
#    variablecmd = ""
#    for s in range(len(MFPs)):
#        variablecmd = " sigmaT=" + str(float(MFPs[s])/0.030) + \
#                      " outFilePrefix=" + outFilePrefix + "MFPs_" + MFPs[s] 
#        cmd = globalcmd + variablecmd
#        if printCMDs:
#            print(cmd)
#        if executeCMDs:
#            if AWS:
#                os.system("echo " + cmd + " > temp.sh")
#                os.system("qsub temp.sh ")
#            else:
#                os.system(cmd)
#
#del MFPs
#
## 3. Albedos
#albedos = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.99"]
#for r in range(startIndex, startIndex + runs):
#    variablecmd = ""
#    for s in range(len(albedos)):
#        variablecmd = " albedo=" + albedos[s] + \
#                      " outFilePrefix=" + outFilePrefix + "albedo_" + albedos[s] 
#        cmd = globalcmd + variablecmd
#        if printCMDs:
#            print(cmd)
#        if executeCMDs:
#            if AWS:
#                os.system("echo " + cmd + " > temp.sh")
#                os.system("qsub temp.sh ")
#            else:
#                os.system(cmd)
#
#del albedos
#
## 4. gs
#gs = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.99"]
#for r in range(startIndex, startIndex + runs):
#    variablecmd = ""
#    for s in range(len(gs)):
#        variablecmd = " gVal=" + gs[s] + \
#                      " outFilePrefix=" + outFilePrefix + "g_" + gs[s] 
#        cmd = globalcmd + variablecmd
#        if printCMDs:
#            print(cmd)
#        if executeCMDs:
#            if AWS:
#                os.system("echo " + cmd + " > temp.sh")
#                os.system("qsub temp.sh ")
#            else:
#                os.system(cmd)
#
#del gs
#
## 5. freqs
#freqs = ["802000", "812000", "822000", "832000", "842000", "852000", "862000", "872000", "882000", "892000", "902000", "912000", "922000", "932000", "942000", "952000", "962000", "972000", "982000", "992000", "1002000"] 
#n_maxs = ["0.00069", "0.00067", "0.00066", "0.00064", "0.00063", "0.00061", "0.00060", "0.00059", "0.00057", "0.00056", "0.00055", "0.00054", "0.00053", "0.00052", "0.00051", "0.00050", "0.00049", "0.00048", "0.00047", "0.00046", "0.00045"]
##n_maxs = []
#for r in range(startIndex, startIndex + runs):
#    variablecmd = ""
#    for s in range(len(freqs)):
#        variablecmd = " f_u=" + freqs[s] + \
#                      " n_max=" + n_maxs[s] + \
#                      " outFilePrefix=" + outFilePrefix + "freq_" + freqs[s] 
#        cmd = globalcmd + variablecmd
#        if printCMDs:
#            print(cmd)
#        if executeCMDs:
#            if AWS:
#                os.system("echo " + cmd + " > temp.sh")
#                os.system("qsub temp.sh ")
#            else:
#                os.system(cmd)
#
#del freqs
#del n_maxs
#
#
## 6. amplitudes
#
#n_maxs = ["0.000657", ".0059", ".0161", ".0315", ".0521", ".0782"]
#
#for r in range(startIndex, startIndex + runs):
#    variablecmd = ""
#    for s in range(len(n_maxs)):
#        variablecmd = " n_max=" + n_maxs[s] + \
#                      " outFilePrefix=" + outFilePrefix + "n_max_" + n_maxs[s] 
#        cmd = globalcmd + variablecmd
#        if printCMDs:
#            print(cmd)
#        if executeCMDs:
#            if AWS:
#                os.system("echo " + cmd + " > temp.sh")
#                os.system("qsub temp.sh ")
#            else:
#                os.system(cmd)
#
## 8. Ratio of beams
#
#emitter_sizes = [".00025", ".0005", ".001", ".002", ".004"]
#
#for r in range(startIndex, startIndex + runs):
#    variablecmd = ""
#    for s in range(len(emitter_sizes)):
#        variablecmd = " emitter_size=" + emitter_sizes[s] + \
#                      " outFilePrefix=" + outFilePrefix + "ratio_" + str(float(emitter_sizes[s])/.002)
#        cmd = globalcmd + variablecmd
#        if printCMDs:
#            print(cmd)
#        if executeCMDs:
#            if AWS:
#                os.system("echo " + cmd + " > temp.sh")
#                os.system("qsub temp.sh ")
#            else:
#                os.system(cmd)
#
# 9. pulse widths

#pulsewidths = ["1", "2", "5", "10", "25", "50", "75", "100"]
pulsewidths = ["0", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100"]

for r in range(startIndex, startIndex + runs):
    variablecmd = ""
    for s in range(len(pulsewidths)):
        phi_min = 1.5708 - 3.1416*float(pulsewidths[s])/100
        phi_max = 1.5708 + 3.1416*float(pulsewidths[s])/100
        variablecmd = " phi_min=" + str(phi_min) + \
                      " phi_max=" + str(phi_max) + \
                      " outFilePrefix=" + outFilePrefix + "pulsed_" + pulsewidths[s]
        cmd = globalcmd + variablecmd
        if printCMDs:
            print(cmd)
        if executeCMDs:
            if AWS:
                os.system("echo " + cmd + " > temp.sh")
                os.system("qsub temp.sh ")
            else:
                os.system(cmd)

