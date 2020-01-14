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
from mrc2jpg_p import mrc2jpg, star2miclist
import multiprocessing as mp
import shutil
import pandas as pd
import sys

def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help="Input directory of the micrographs in mrc format. Cannot contain other directories inside (excluding directories made by MicAssess).")
    ap.add_argument('-m', '--model', default='./models/micassess_051419.h5',
                    help='Path to the model.h5 file.')
    ap.add_argument('-o', '--output', default='good_micrographs.star',
                    help="Name of the output star file. Default is good_micrographs.star.")
    ap.add_argument('-b', '--batch_size', type=int, default=32,
                    help="Batch size used in prediction. Default is 32. If memory error/warning appears, try lower this number to 16, 8, or even lower.")
    ap.add_argument('-t', '--threshold', type=float, default=0.1,
                    help="Threshold for classification. Default is 0.1. Higher number will cause more good micrographs being classified as bad.")
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

def star2df(starfile):
    with open(starfile) as f:
        star = f.readlines()

    for i in range(len(star)):
        if 'loop_' in star[i]:
            start_idx = i
            break
    key_idx = []
    for j in range(start_idx+1, len(star)):
        if star[j].startswith('_'):
            key_idx.append(j)

    keys = [star[ii] for ii in key_idx]
    star_df = star[1+key_idx[-1]:]
    star_df = [x.split() for x in star_df]
    star_df = pd.DataFrame(star_df)
    star_df = star_df.dropna()
    star_df.columns = keys

    return star_df

def df2star(star_df, star_name):
    header = ['data_ \n', '\n', 'loop_ \n']
    keys = star_df.columns.tolist()

    with open(star_name, 'w') as f:
        for l_0 in header:
            f.write(l_0)
        for l_1 in keys:
            f.write(l_1)
        for i in range(len(star_df)):
            s = '  '.join(star_df.iloc[i].tolist())
            f.write(s + ' \n')

def predict(**args):
    start_dir = os.getcwd()
    print('Start to assess micrographs with MicAssess.')
    model = load_model(args['model'])
    batch_size = args['batch_size']
    test_data_dir = os.path.join(os.path.abspath(os.path.join(args['input'], os.pardir)), 'MicAssess') # MicAssess is in the par dir of input file
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
    shutil.rmtree('data') # after prediction, remove the data directory

    # write the output file
    os.chdir(start_dir)
    os.chdir(os.path.abspath(os.path.dirname(args['input']))) # navigate to the par dir of input file
    try:
        os.remove(args['output'])
    except OSError:
        pass
    star_df = star2df(os.path.basename(args['input']))
    goodlist_base = [os.path.basename(f)[:-4] for f in goodlist]
    badindex = []
    for i in range(len(star_df)):
        if os.path.basename(star_df['_rlnMicrographName\n'][i])[:-4] not in goodlist_base:
            badindex.append(i)
    new_star_df = star_df.drop(badindex)
    df2star(new_star_df, args['output'])
    # with open('goodlist', 'wb') as f:
    #     pickle.dump(goodlist, f)

    print('All finished!')

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    start_dir = os.getcwd()
    args = setupParserOptions()
    os.chdir(start_dir)
    mrc2jpg(**args)
    os.chdir(start_dir)
    predict(**args)
