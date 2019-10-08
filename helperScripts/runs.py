#!/usr/bin/python
import os
from colorama import Fore, Back, Style

# Renderer
renderer      = "/home/ubuntu/AOOCT_V1/renderer/samples/renderer3d_sample_bin"
outFolder     = "/home/ubuntu/AOOCT_V1/renderings"
outFilePrefix = "RRTE_complexTiming_noAreaSource"
clusterTemp   = "temp.sh"
printcmds     = True
submitcmds    = True


# Renderer options
numpackets    = 20 # Number of parallel cores
numPhotons    = 1e6 # Number of samples for each job
useDirect     = "true"
sigmaT        = 1
albedo        = .9
gVal          = 0
tBins         = 128

pathLengthMax = 16;
# US options
n_max         = 0 

# Merge the packets in a single rendering
MergeExecutable         = "/home/ubuntu/AOOCT_V1/helperScripts/mergeMultipleRenderings"
tempMergeExecutable     = "tempMergeCommands_useDirect_nMax_.001.sh"
deleteIntermediateRuns  = True # include deletion of individul runs? Not coded yet


open(tempMergeExecutable, 'w').close() # Clear this file first
                
os.system("echo \"" + MergeExecutable + " prefix=" + outFolder + "/" + outFilePrefix + "_" + str(numPhotons) + "_" + str(n_max) + "_" + useDirect + \
                                        " renderings="  + str(numpackets) + \
                                        " tBins=" + str(tBins) + \
                                        "\" >> " + tempMergeExecutable)

for i in range(0, numpackets):
    fileName = outFolder + "/" + outFilePrefix + "_" + str(numPhotons) + "_" + str(n_max) + "_" + str(i) 
    cmd = "time " + renderer + " numPhotons=" + str(numPhotons) + \
                     " outFilePrefix="  + fileName + \
                     " useDirect="  + useDirect + \
                     " sigmaT="     + str(sigmaT) + \
                     " albedo="     + str(albedo) + \
                     " gVal="       + str(gVal) + \
                     " n_max="       + str(n_max) + \
                     " pathLengthMax=" + str(pathLengthMax) + \
                     " pathLengthBins=" + str(tBins) 
    
    os.system("echo \"" + cmd + "\"" + " > " + clusterTemp)
    if printcmds:
        os.system("cat " + clusterTemp)
    if submitcmds:
        os.system("qsub " + clusterTemp)
    if deleteIntermediateRuns:
        for j in range(0,tBins):
            os.system("echo \"" + "rm " + fileName + "_" + str(j) + ".pfm" + " \" >> " + tempMergeExecutable)

#if deleteIntermediateRuns:
#    print(Back.RED + "deleteIntermediateRuns not implemented yet")

#if printcmds:
#    print(Fore.GREEN + "printing the mergeExecutable:")
#    os.system("cat " + tempMergeExecutable)
