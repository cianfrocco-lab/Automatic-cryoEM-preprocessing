# -*- coding: utf-8 -*-
"""
Check if the object in the 2d class average is centered, and if there are
multiple objects in the class average image.
Use saliency map.
"""

from PIL import Image
import numpy as np
from scipy import ndimage
from skimage import measure
import cv2
import matplotlib.pyplot as plt

def check_center(img_name):
    img = Image.open(img_name)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(np.asarray(img))

    saliencyMap_bin = np.where(saliencyMap > 0.2, 1, 0)
    kernel = np.ones((5,5),np.uint8)
    saliencyMap_bin = cv2.morphologyEx(np.float32(saliencyMap_bin), cv2.MORPH_CLOSE, kernel)

    aa = measure.label(saliencyMap_bin, background=0)
    propsa = measure.regionprops(aa)

    max_area = propsa[0].area
    for label in propsa:
        if label.area > max_area:
            max_area = label.area

    object_num = 0
    for label in propsa:
        if label.area > max(0.5*max_area, 40):
            object_num += 1

    centerness = np.divide(ndimage.measurements.center_of_mass(saliencyMap_bin), saliencyMap_bin.shape)

    if abs(centerness[0]-0.5) > 0.15 or abs(centerness[1]-0.5) > 0.15 or object_num > 1:
        return False
    else:
        return True
