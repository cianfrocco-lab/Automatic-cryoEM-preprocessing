#!/usr/bin/env python3
'''
Use the 2DAssess to assess class averages.
Will predict each image files in the input directory and save to the folders
with corresponding labels in the input directory path.
Input should be the mrcs file of the 2D class averages.
Will also save a goodlist file (pickle file) for future use.
'''

from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras.optimizers import *
from keras import backend as keras
import numpy as np
import os
import argparse
from shutil import copy2
import shutil
import glob
import pickle
from functools import partial, update_wrapper
from itertools import product
from classavg_preprocessing_p import preprocess
from check_center_p import check_center
from classavg2jpg_p import save_mrcs
import re

def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help="Input mrcs file of 2D class averages.")
    ap.add_argument('-m', '--model', default='./models/2dassess_062119.h5',
                    help='Path to the model.h5 file.')
    ap.add_argument('-n', '--name', default='particle',
                    help="Name of the particle. Default is particle.")
    ap.add_argument('-o', '--output', default='2DAssess',
                    help="Name of the output directory. Default is 2DAssess.")
    args = vars(ap.parse_args())
    return args

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

def predict(**args):
    print('Assessing 2D class averages with 2DAssess....')
    test_data_dir = os.path.abspath(args['output'])
    labels = ['Clip', 'Edge', 'Good', 'Noise']
    os.chdir(test_data_dir)
    for l in labels:
        shutil.rmtree(l, ignore_errors=True)
    w_array = np.ones((4, 4))
    w_array[(0,1,3), 2] = 1.0
    w_array[2, (0,1,3)] = 1.0
    ncce = wrapped_partial(w_categorical_crossentropy, weights=w_array)

    model = load_model(args['model'], custom_objects={'w_categorical_crossentropy': wrapped_partial(w_categorical_crossentropy, weights=w_array)})
    model.compile(optimizer = Adam(lr = 1e-4), loss = ncce, metrics = [metrics.categorical_accuracy])

    test_datagen = ImageDataGenerator(
        rescale = 1./255,
        samplewise_center=True,
        samplewise_std_normalization=True)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        shuffle=False,
        target_size=(256, 256),
        batch_size=32,
        color_mode='grayscale',
        class_mode=None,
        interpolation='lanczos')
    prob = model.predict_generator(test_generator)
    print('Assessment finished. Copying files to corresponding directories....')

    for l in labels:
        os.mkdir(l)
    i = 0
    for file in sorted(glob.glob('data/*.jpg')):
        if labels[np.argmax(prob[i])] == 'good':
            if check_center(file) == True:
                copy2(file, 'Good')
            else:
                copy2(file, 'Clipping')
        else:
            copy2(file, labels[np.argmax(prob[i])])
        i = i + 1

    shutil.rmtree('data') # after prediction, remove the data directory
    good_idx = []
    for fname in os.listdir('Good'):
        good_idx.append(re.findall((args['name']+'_'+'(\d+)'), fname[:-4])[0])

    print('All finished! Outputs are stored in', test_data_dir)
    print('Good class averages indices are (starting from 1): ', end='')
    print(', '.join(good_idx))

if __name__ == '__main__':
    start_dir = os.getcwd()
    args = setupParserOptions()
    args['model'] = os.path.abspath(args['model'])
    os.chdir(start_dir)
    save_mrcs(**args)
    predict(**args)
