#!/usr/bin/python
import os
import sys
import numpy as np
import subprocess
from colorama import Fore, Back, Style

# Renderer
renderer      = "/home/ubuntu/AOOCT_V1/renderer/samples/renderer3d_sample_bin"
outFolder     = "/home/ubuntu/AOOCT_V1/renderings"
prefix        = "Projector"
clusterTemp   = "temp.sh"
printcmds     = False
submitcmds    = True
eMailID       = "aditya.eee.nitw@gmail.com"

# Renderer options
numpackets    = 20 # Number of parallel cores
numPhotons    = 1e6 # Number of samples for each job
sigmaT        = 0
albedo        = .9
gVal          = 0
n_o           = 1.3333
n_max         = 0
mode          = 0
projectorTexture = "/home/ubuntu/AOOCT_V1/renderer/images/White.pfm"
useDirect     = "true"
maxDepth      = -1
maxPathlength = -1
pathLengthMin = 0
pathLengthMax = 64
pathLengthBins= 128
spatialX      = 128
spatialY      = 128
emitter_sensor_size = 0.01

# Read input if available
for i in range(1, len(sys.argv)):
    param = sys.argv[i].split("=")
    if (str(param[0]) == "renderer"):
        renderer = param[1]
    elif (str(param[0]) == "outFolder"):
        outFolder = param[1]
    elif (str(param[0]) == "prefix"):
        prefix = param[1]
    elif (str(param[0]) == "clusterTemp"):
        clusterTemp = param[1]
    elif (str(param[0]) == "printcmds"):
        printcmds = param[1]
    elif (str(param[0]) == "submitcmds"):
        submitcmds = param[1]
    elif (str(param[0]) == "numpackets"):
        numpackets = int(param[1])
    elif (str(param[0]) == "numPhotons"):
        numPhotons = int(param[1])  
    elif (str(param[0]) == "sigmaT"):
        sigmaT = float(param[1])
    elif (str(param[0]) == "albedo"):
        albedo = float(param[1])
    elif (str(param[0]) == "gVal"):
        gVal = float(param[1])
    elif (str(param[0]) == "n_o"):
        n_o = float(param[1])
    elif (str(param[0]) == "n_max"):
        n_max = float(param[1])
    elif (str(param[0]) == "projectorTexture"):
        projectorTexture = param[1]
    elif (str(param[0]) == "mode"):
        mode = int(param[1])
    elif (str(param[0]) == "useDirect"):
        useDirect = param[1]
    elif (str(param[0]) == "maxDepth"):
        maxDepth = int(param[1])
    elif (str(param[0]) == "maxPathlength"):
        maxPathlength = float(param[1])
    elif (str(param[0]) == "pathLengthMin"):
        pathLengthMin = float(param[1])
    elif (str(param[0]) == "pathLengthMax"):
        pathLengthMax = float(param[1])
    elif (str(param[0]) == "pathLengthBins"):
        pathLengthBins = float(param[1])
    elif (str(param[0]) == "spatialX"):
        spatialX = int(param[1])
    elif (str(param[0]) == "spatialY"):
        spatialY = int(param[1])
    elif (str(param[0]) == "emitter_sensor_size"):
        emitter_sensor_size = float(param[1])
    else:
        print("Unknown variable in the input argument:" + \
		        "Should be one of " + \
				"outFolder, " + \
				"prefix, " + \
				"clusterTemp, " + \
				"printcmds, " + \
				"submitcmds, " + \
				"numpackets, " + \
				"numPhotons, " + \
				"sigmaT, " + \
				"albedo, " + \
				"gVal, " + \
				"n_o, " + \
				"n_max, " + \
				"mode, " + \
				"useDirect, " + \
				"maxDepth, " + \
				"maxPathlength, " + \
				"pathLengthMin, " + \
				"pathLengthMax, " + \
				"pathLengthBins, " + \
				"spatialX, " + \
				"spatialY, " + \
				"emitter_sensor_size" + \
				"")
        sys.exit()

# suffix for both tempMerge and submit common
suffix        = "_sca_" + str(sigmaT) + "_" + str(albedo) + "_" + str(gVal) + \
                "_n_" + str(n_o) + "_" + str(n_max) + \
                "_" + useDirect + \
                "_path_" + str(pathLengthMin) + "_" + str(pathLengthMax) + "_" + str(pathLengthBins) 

#Final file name prefix
fileNamePrefix = outFolder + "/" + prefix + suffix

# Merge the packets in a single rendering
MergeExecutable         = "/home/ubuntu/AOOCT_V1/helperScripts/mergeMultipleRenderings"
tempMergeExecutable     = "tempMergeCommands/Merge_" + str(numpackets*numPhotons) + suffix + ".sh" 
deleteIntermediateRuns  = True # include deletion of individul runs

open(tempMergeExecutable, 'w').close() # Clear this file first
                
os.system("echo \"" + MergeExecutable + " prefix=" + fileNamePrefix + \
                                        " renderings="  + str(numpackets) + \
                                        " pathLengthBins=" + str(pathLengthBins) + \
                                        "\" >> " + tempMergeExecutable)

dependentJobs = ""
for i in range(0, numpackets):
    fileName = fileNamePrefix + "_" + str(i) 
    cmd = "time " + renderer + \
                    " outFilePrefix="  + fileName + \
					" numPhotons=" + str(numPhotons) + \
					" sigmaT=" + str(sigmaT) + \
					" albedo=" + str(albedo) + \
					" gVal=" + str(gVal) + \
					" n_o=" + str(n_o) + \
					" n_max=" + str(n_max) + \
					" mode=" + str(mode) + \
					" projectorTexture=" + str(projectorTexture) + \
					" useDirect=" + str(useDirect) + \
					" maxDepth=" + str(maxDepth) + \
					" maxPathlength=" + str(maxPathlength) + \
					" pathLengthMin=" + str(pathLengthMin) + \
					" pathLengthMax=" + str(pathLengthMax) + \
					" pathLengthBins=" + str(pathLengthBins) 
    
    os.system("echo \"" + cmd + "\"" + " > " + clusterTemp)
    if printcmds:
        os.system("cat " + clusterTemp)
    if submitcmds:
        temp = subprocess.check_output("qsub " + clusterTemp, shell=True);
        jobid = [int(s) for s in temp.split() if s.isdigit()]
        dependentJobs += str(jobid[0]) + ","
    if deleteIntermediateRuns:
        for j in range(0,pathLengthBins):
            os.system("echo \"" + "rm " + fileName + "_" + str(j) + ".pfm" + " \" >> " + tempMergeExecutable)
dependentJobs = dependentJobs[:-1]


if printcmds:
    print(Fore.GREEN + "The mergeExecutable script is in:" + tempMergeExecutable)

if submitcmds:
    os.system("qsub -hold_jid " + dependentJobs + " " + tempMergeExecutable)
