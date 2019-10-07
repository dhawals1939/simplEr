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
numPhotons    = 1e5 # Number of samples for each job
useDirect     = "false"
sigmaT        = .01
albedo        = .9
gVal          = 0
tBins         = 128

# Merge the packets in a single rendering
MergeExecutable         = "/home/ubuntu/AOOCT_V1/helperScripts/mergeMultipleRenderings"
tempMergeExecutable     = "tempMergeCommands.sh"
deleteIntermediateRuns  = False # include deletion of individul runs? Not coded yet


open(tempMergeExecutable, 'w').close() # Clear this file first
                

for i in range(0, numpackets):
    cmd = renderer + " numPhotons=" + str(numPhotons) + \
                     " outFilePrefix="  + outFolder + "/" + outFilePrefix + "_" + str(i) + "_" + str(numPhotons) + ".pfm" \
                     " useDirect="  + useDirect + \
                     " sigmaT="     + str(sigmaT) + \
                     " albedo="     + str(albedo) + \
                     " gVal="       + str(gVal) + \
                     " pathLengthBins=" + str(tBins) 
    
    os.system("echo \"" + cmd + "\"" + " > " + clusterTemp)
    if printcmds:
        os.system("cat " + clusterTemp)
    if submitcmds:
        os.system("qsub " + clusterTemp)

os.system("echo \"" + MergeExecutable + " prefix=" + outFolder + "/" + outFilePrefix  + \
                                        " renderings="  + str(numpackets) + \
                                        " tBins=" + str(tBins) + \
                                        "\" >>" + tempMergeExecutable)
if deleteIntermediateRuns:
    print(Back.RED + "deleteIntermediateRuns not implemented yet")

if printcmds:
    print(Fore.GREEN + "printing the mergeExecutable:")
    os.system("cat " + tempMergeExecutable)
