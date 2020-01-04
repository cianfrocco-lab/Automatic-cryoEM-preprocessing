'''
Preprocessing of the image before feeding into the CNN.
'''

import numpy as np
from scipy import ndimage
import pandas as pd

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def mask_img(temp_img):
    mask = create_circular_mask(temp_img.shape[0], temp_img.shape[1])
    masked_img = temp_img.copy()
    masked_img[~mask] = 0
    return masked_img

def preprocess(img):
    '''
    Center to 0 and divide by std to normalize.
    And then apply a circular mask to make it rotatable.
    '''
    norm_img = (img - np.mean(img))/np.std(img)
    masked_img = mask_img(norm_img)
    return masked_img
