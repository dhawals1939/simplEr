
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import time

import sys
import os
this_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_dir + '/../bin')
from random import seed
from random import random
from functools import partial


from multiprocessing import Pool, TimeoutError
import image_pybind

def readAllFrequencies(nSearches, n_max_min, n_max_max, f_u_min, f_u_max, MFPs, s, folderPrefix, Mask, v_n):
    Intensities = np.zeros((nSearches))
    for f_n in range(0, nSearches):
        n_max = round(n_max_min + v_n*(n_max_max - n_max_min)/nSearches, 4)
        f_u = round(f_u_min + f_n*(f_u_max - f_u_min)/nSearches,1)
        outFileName = folderPrefix + "MFPs_" + MFPs[s] + "_n_max_" + str(n_max) + "_f_u_" + str(f_u) + ".pfm3d"
        if not os.path.exists(outFileName):
            print(outFileName + " does not exist")
            exit()
        Image = image_pybind.SmallImage(10,10,10)
        Image.readPFM3D(outFileName)
        temp = 0.0
        for i in range(0, 1023):
            for j in range(0, 1023):
                temp += Mask[i,j] * Image.getPixel(i, j, 0)
        Intensities[f_n] = temp;
    return Intensities

#fileName = '/home/apedired/Adi2TB/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr_angletracingNEE/renderer/images/Disk_501.pfm'
pool = Pool()
folderPrefix = '/shared/LSA/VF/1M/ER_'

MFPs = ["0"];
n_max_min = .0001;
n_max_max = .10;
f_u_min = 277000;
f_u_max = 2496000;
nSearches = 100;

#Image.readPFM3D(fileName)
#print(type(Image))
#print(Image.getPixel(538, 534, 0))

sensor_size = 5;
object_radius = .1; # photons on 2 cm focus
radius = object_radius/sensor_size*1024;

Mask = np.zeros((1024,1024));
Mask1Range_c = 511.5;
for i in range(0, 1023):
    for j in range(0, 1023):
        if( (i-Mask1Range_c)**2 + (j-Mask1Range_c)**2 <= radius**2):
            Mask[i,j] = 1

Intensities = np.zeros((len(MFPs), nSearches, nSearches))

for s in range(0, len(MFPs)):
    func = partial(readAllFrequencies, nSearches, n_max_min, n_max_max, f_u_min, f_u_max, MFPs, s, folderPrefix, Mask)
    for v_n, res in enumerate(pool.imap(func, range(0, nSearches)), 0):
        Intensities[s, v_n, :] = res



m_dic = {"Intensities": Intensities, "label": "Radius_0.1"}

savemat("wrapSrinivasData.mat", m_dic)
            


#
