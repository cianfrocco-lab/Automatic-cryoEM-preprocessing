'''
Helper functions for simple image preprocessing used.
'''
import numpy as np
from PIL import Image
from scipy import ndimage
import pandas as pd
# from PIL import ImageOps


def createCircularMask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def maskImg(img):
    mask = createCircularMask(img.shape[0], img.shape[1])
    masked_img = img.copy()
    masked_img[~mask] = 0
    return masked_img

#### BELOW: for micrographs
def downsample(img, height=494):
    '''
    Downsample 2d array using fourier transform.
    factor is the downsample factor.
    '''
    m,n = img.shape[-2:]
    ds_factor = m/height
    # height = round(m/ds_factor/2)*2
    width = round(n/ds_factor/2)*2
    F = np.fft.rfft2(img)
    A = F[...,0:height//2,0:width//2+1]
    B = F[...,-height//2:,0:width//2+1]
    F = np.concatenate([A, B], axis=0)
    f = np.fft.irfft2(F, s=(height, width))
    return f

def scaleImage(img, height=494):
    '''
    Downsample image, scale the pixel value from 0-255 and save it as the Image object.
    '''
    new_img = downsample(img, height)
    new_img = ((img-img.min())/((img.max()-img.min())+1e-7)*255).astype('uint8')
    new_img = Image.fromarray(new_img)
    new_img = new_img.convert("L")
    return new_img

def cropLeft(img, cropx, cropy):
    y = img.shape[0]
    startx = 0
    starty = y//2-(cropy//2)
    new_img_left = img[starty:starty+cropy,startx:startx+cropx]
    new_img_left = Image.fromarray(new_img_left)
    new_img_left = new_img_left.convert("L")
    return new_img_left

def cropRight(img, cropx, cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = x-cropx
    starty = y//2-(cropy//2)
    new_img_right = img[starty:starty+cropy,startx:startx+cropx]
    new_img_right = Image.fromarray(new_img_right)
    new_img_right = new_img_right.convert("L")
    return new_img_right

def cropCenter(img,cropx,cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def preprocessMics(img):
    '''
    Crop the images to make it square.
    Center to 0 and divide by std to normalize.
    And then apply a circular mask to make it rotatable.
    '''
    short_edge = min(img.shape[0], img.shape[1])
    square_img = cropCenter(img, short_edge, short_edge)
    norm_img = (square_img - np.mean(square_img))/np.std(square_img)
    masked_img = maskImg(norm_img)
    return masked_img

#### BELOW: for class averages
def cutByRadius(img):
    '''
    Crop the images (2d class averages) by the radius of the mask.
    Will find the radius from the image and crop the image.
    '''
    h = img.shape[0]
    w = img.shape[1]
    # empty_val = img[0,0] # because the image is already masked (2d class avg), the [0,0] point must be empty
    edge_l = 0
    for i in range(w):
        if np.sum(img[i,:]) > 1e-7 or np.sum(img[:,i]) < -1e-7:
            edge_l = i
            break
    edge_r = 0
    for ii in range(w):
        if np.sum(img[-ii,:]) > 1e-7 or np.sum(img[:,-ii]) < -1e-7:
            edge_r = ii
            break
    edge_t = 0
    for j in range(h):
        if np.sum(img[:,j]) > 1e-7 or np.sum(img[:,j]) < -1e-7:
            edge_t = j
            break
    edge_b = 0
    for jj in range(h):
        if np.sum(img[:,-jj]) > 1e-7 or np.sum(img[:,-jj]) < -1e-7:
            edge_b = jj
            break
    edge = min(edge_l, edge_r, edge_t, edge_b)
    new_img = img[edge:h-edge+1, edge:w-edge+1]
    return new_img


def preprocessClsavg(img):
    '''
    Center to 0 and divide by std to normalize.
    And then apply a circular mask to make it rotatable.
    '''
    norm_img = (img - np.mean(img))/np.std(img + 0.00001)
    masked_img = maskImg(norm_img)
    return masked_img
