#!/usr/bin/env python3
'''
Use MicAssess to predict micrographs.
Will predict each image files in the input directory as good or bad micrographs,
and save to the 'good' and 'bad' folders in the input directory path.
Will also save a goodlist file (pickle file) for future use.

INPUTS: Input directory of the micrographs in jpg format.
        Path to the .h5 model file.

To use: python micassess.py -i <input_path> -m <model_path>
'''

from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras import backend as keras
import numpy as np
import os
import argparse
from shutil import copy2
import glob
import pickle
from mrc2jpg_p import mrc2jpg
import multiprocessing as mp
import shutil

def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help="Input directory of the micrographs in mrc format. Cannot contain other directories inside (excluding directories made by MicAssess).")
    ap.add_argument('-m', '--model',
                    help='Path to the model.h5 file.')
    ap.add_argument('-b', '--batch_size', type=int, default=32,
                    help="Batch size used in prediction.")
    ap.add_argument('-t', '--threshold', type=float, default=0.1,
                    help="Threshold for classification.")
    args = vars(ap.parse_args())
    return args

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

def crop_center(img,cropx,cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def preprocess(img):
    '''
    Crop the images to make it square.
    Center to 0 and divide by std to normalize.
    And then apply a circular mask to make it rotatable.
    '''
    short_edge = min(img.shape[0], img.shape[1])
    square_img = crop_center(img, short_edge, short_edge)
    norm_img = (square_img - np.mean(square_img))/np.std(square_img)
    masked_img = mask_img(norm_img)
    return masked_img

def copygoodfile(file):
    copy2(file, 'pred_good')

def copybadfile(file):
    copy2(file, 'pred_bad')

#%%
def predict(**args):
    print('Start to assess micrographs with MicAssess.')
    model = load_model(args['model'])
    batch_size = args['batch_size']
    test_data_dir = args['input']
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess)

    test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(494, 494), batch_size=batch_size, color_mode='grayscale', class_mode=None, shuffle=False)
    prob = model.predict_generator(test_generator)
    print('Assessment finished. Copying files to good and bad directories....')
    os.chdir(test_data_dir)
    os.mkdir('pred_good')
    os.mkdir('pred_bad')

    good_idx = np.where(prob > args['threshold'])[0]
    bad_idx = np.where(prob <= args['threshold'])[0]
    goodlist = list(sorted(glob.glob('data/*.jpg')[i] for i in good_idx))
    badlist = list(sorted(glob.glob('data/*.jpg')[i] for i in bad_idx))

    pool = mp.Pool(mp.cpu_count())
    pool.map(copygoodfile, [file for file in goodlist])
    pool.map(copybadfile, [file for file in badlist])
    pool.close()

    with open('goodlist', 'wb') as f:
        pickle.dump(goodlist, f)

    shutil.rmtree('data') # after prediction, remove the data directory
    print('All finished! Outputs are stored in the input directory.')

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    args = setupParserOptions()
    mrc2jpg(**args)
    predict(**args)
