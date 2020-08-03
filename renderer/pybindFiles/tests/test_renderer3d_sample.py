
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
this_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_dir + '/../bin')

import image_pybind

fileName = '/home/apedired/Adi2TB/Dropbox/AccoustoOptics+InvRendering/CodeEtc/SkeletalRenderer/ercrdr_angletracingNEE/renderer/images/Disk_501.pfm'

Image = image_pybind.SmallImage(10,10)
Image.readFile(fileName, Image.EFileFormat.EPFM)
print(type(Image))
print(Image.getXRes())
print(Image.getYRes())
print(Image.getPixel(1, 1))
print(Image.getPixel(73, 73))

