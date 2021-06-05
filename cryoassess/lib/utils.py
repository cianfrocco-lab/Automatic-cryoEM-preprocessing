import numpy as np
from functools import partial, update_wrapper
from itertools import product
from tensorflow.keras import backend as K


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_true, y_pred) * final_mask


def ncce(w_categorical_crossentropy):
    w_array = np.ones((4, 4))
    w_array[(0,1,3), 2] = 1.0
    w_array[2, (0,1,3)] = 1.0
    ncce = wrapped_partial(w_categorical_crossentropy, weights=w_array)
    return ncce


def crop(img, cropx, cropy, position):

    y = img.shape[0]
    x = img.shape[1]

    if position == 'center':
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
    elif position == 'left':
        startx = 0
        starty = y//2-(cropy//2)
    elif position == 'right':
        startx = x-cropx
        starty = y//2-(cropy//2)

    return img[starty:starty+cropy,startx:startx+cropx]

def normalize(x):
    # x /= 127.5
    # x -= 1.
    x = (x - np.mean(x)) / np.std(x)
    return x


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


def mask_img(temp_img, center=None, radius=None):
    mask = create_circular_mask(temp_img.shape[0], temp_img.shape[1], center=None, radius=None)
    masked_img = temp_img.copy()
    masked_img[~mask] = 0
    return masked_img


def preprocess_c(img):
    '''
    Crop the images to make it square.
    Normalize the image from -1 to 1.
    And then apply a circular mask to make it rotatable.
    '''
    # short_edge = min(img.shape[0], img.shape[1])
    # square_img = crop(img, short_edge, short_edge, position='center')
    norm_img = normalize(img)
    # norm_img = normalize(square_img)
    masked_img = mask_img(norm_img)

    return masked_img

def preprocess_l(img):

    h = img.shape[0]
    w = img.shape[1]
    # short_edge = min(img.shape[0], img.shape[1])
    # square_img = crop(img, short_edge, short_edge, position='center')
    norm_img = normalize(img)
    # norm_img = normalize(square_img)
    masked_img = mask_img(norm_img, center=[int(h/2), int(h/2)])

    return masked_img

def preprocess_r(img):

    h = img.shape[0]
    w = img.shape[1]
    # short_edge = min(img.shape[0], img.shape[1])
    # square_img = crop(img, short_edge, short_edge, position='center')
    norm_img = normalize(img)
    # norm_img = normalize(square_img)
    masked_img = mask_img(norm_img, center=[int(w-h/2), int(h/2)])

    return masked_img
