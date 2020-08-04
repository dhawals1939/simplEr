
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
this_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_dir + '/../bin')

import image_pybind

#fileName = '/home/apedired/Adi2TB/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr_angletracingNEE/renderer/images/Disk_501.pfm'
fileName = '/home/apedired/Adi2TB/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr_angletracingNEE/renderer/pybindFiles/ER_MFPs_0_n_max_0.0001_f_u_1253360.0.pfm3d'

Image = image_pybind.SmallImage(10,10,10)
Image.readPFM3D(fileName)
print(type(Image))
print(Image.getPixel(538, 534, 0))
#
