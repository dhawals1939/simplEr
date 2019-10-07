#!/usr/bin/python
import os

executable = "/home/ubuntu/mts_installs/mitsuba_transient/helperScripts/MeanAndVarianceEXRs/createMeanAndVarianceImages"
folder     = "/shared/LATCGroundTruth"

bunnies    = [ "908", 
               "3632", 
               "8172", 
               "14528", 
               "22700", 
               "44492",
               "90800",
               "153452",
               "204300"
               ]
samples    = [ 10,   8,   6,   5,   4,  2, 1, 1, 1]
ss         = [100, 110, 125, 113, 110, 250, 450, 500, 110]
#bunnies    = [ 
#               "153452",
#               "204300"
#               ]
#samples    = [1, 1]
#ss         = [500, 110]


for j in range(0, len(bunnies)):
    for i in range(75, 101):
        cmd = "mitsuba -D adapSampling=false -D ldSampling=false -D samples=787 -D tMin=1704 -D tMax=1708 -D tRes=4 -D subSamples=1 -D forceBounce=false -D sBounce=0 -D tBounce=0 -D decomposition=transient ~/BlenderScenes/cbox/cbox_bunnies.xml -D bunny=meshes/bunnies/bunnies_" + bunnies[j] + ".obj -o /shared/increasingComplexityToG/cbox_" + bunnies[j] + "_4_t_" + str(i) + ".exr"
        os.system("echo PATH=\"\{\$PATH\}:\/home\/ubuntu\/mitsuba\/dist\" > temp.sh")
        os.system("echo \"" + cmd + "\"" + " >> temp.sh")
        os.system("qsub temp.sh")
        cmd = "mitsuba -D adapSampling=false -D ldSampling=false -D samples=" + str(samples[j]) + " -D tMin=1704 -D tMax=1708 -D tRes=4 -D subSamples=" + str(ss[j]) +  " -D forceBounce=false -D sBounce=0 -D tBounce=0 -D decomposition=transientEllipse ~/BlenderScenes/cbox/cbox_bunnies.xml -D bunny=meshes/bunnies/bunnies_" + bunnies[j] + ".obj -o /shared/increasingComplexityToG/cbox_" + bunnies[j] + "_4_tE_" + str(i) + ".exr"
        os.system("echo PATH=\"\{\$PATH\}:\/home\/ubuntu\/mitsuba\/dist\" > temp.sh")
        os.system("echo \"" + cmd + "\"" + " >> temp.sh")
        os.system("qsub temp.sh")

#cmd = executable + " " + folder + "/DARPA_LATCScene_t_groundTruth 99"
#print cmd
#os.system(cmd)
