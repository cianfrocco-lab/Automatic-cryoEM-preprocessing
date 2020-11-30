#!/usr/bin/env python3
'''
Use MicAssess to predict micrographs.
Will predict each image files in the input directory as good or bad micrographs,
and save to the 'good' and 'bad' folders in the input directory path.
Will also save a goodlist file (pickle file) for future use.

To use: python micassess.py -i <input_path> -m <model_path>
'''

from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
import numpy as np
import os
import argparse
from shutil import copy2
import shutil
import glob
import multiprocessing as mp
import pandas as pd
from cryoassess.mrc2jpg import mrc2jpg
from cryoassess.lib import imgprep
from cryoassess.lib import utils

def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help="Input directory, starfile with a list of micrographs or a pattern "
                         "(<path to a folder>/<pattern>. Pattern could have *, ?, any valid glob wildcard.  "
                         "All of the micrographs must be in mrc format. Cannot contain other directories inside (excluding directories made by MicAssess).",
                    required=True)
    ap.add_argument('-d', '--detector', default='K2',
                    help='K2 or K3 detector?')
    ap.add_argument('-m', '--model', default='./models/micassess_051419.h5',
                    help='Path to the model.h5 file.')
    ap.add_argument('-o', '--output', default='good_micrographs.star',
                    help="Name of the output star file. Default is good_micrographs.star.")
    ap.add_argument('-b', '--batch_size', type=int, default=32,
                    help="Batch size used in prediction. Default is 32. If memory error/warning appears, try lower this number to 16, 8, or even lower.")
    ap.add_argument('-t', '--threshold', type=float, default=0.1,
                    help="Threshold for classification. Default is 0.1. Higher number will cause more good micrographs being classified as bad.")
    ap.add_argument('--threads', type=int, default=None,
                    help='Number of threads for conversion. Default is None, using mp.cpu_count(). If get memory error, set it to a reasonable number.')
    ap.add_argument('--gpus', default='0',
                    help='Specify which gpu(s) to use, e.g. 0,1. Default is 0, which uses only one gpu.')
    args = vars(ap.parse_args())
    return args


def input2star(args):
    input = args['input']
    # if a star file
    if input.endswith('.star'):
        return

    micList = []
    import glob
    input = os.path.basename(input)
    micList = glob.glob(input)

    # Get the dirname
    # folder = os.path.dirname(input)
    # newStarFile = os.path.join(folder, "micA_micrographs.star")
    newStarFile = "micrographs.star"
    print("Generating star file %s" % newStarFile)
    if os.path.exists(newStarFile):
        print("Previous star file found, deleting it.")
        os.remove(newStarFile)
    f = open(newStarFile, "w")
    f.write("data_\n")
    f.write("loop_\n")
    f.write("_rlnMicrographName\n")
    f.writelines('\n'.join(micList))
    f.close()

    args['input'] = newStarFile

def copyGoodFile(file):
    copy2(file, os.path.join('MicAssess','predGood'))

def copyBadFile(file):
    copy2(file, os.path.join('MicAssess','predBad'))

def predict(args):
    start_dir = os.getcwd()
    print('Start to assess micrographs with MicAssess.')
    model = load_model(args['model'])
    detector = args['detector']
    batch_size = args['batch_size']
    input_dir = os.path.abspath(os.path.join(args['input'], os.pardir)) # Directory where the input file is (par dir of input file).
    # os.chdir(input_dir)
    test_data_dir = os.path.join(input_dir, 'MicAssess', 'jpgs') # MicAssess is in the par dir of input file
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    if detector == 'K2':
        test_datagen = ImageDataGenerator(preprocessing_function=imgprep.preprocessMics)
        test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(494, 494), batch_size=batch_size, color_mode='grayscale', class_mode=None, shuffle=False)
        prob = model.predict_generator(test_generator)
        # print(prob)

    if detector == 'K3':
        test_data_dir_left = os.path.join('MicAssess', 'K3Left')
        test_data_dir_right = os.path.join('MicAssess', 'K3Right')
        test_datagen = ImageDataGenerator(preprocessing_function=imgprep.preprocessMics)
        test_generator_left = test_datagen.flow_from_directory(test_data_dir_left, target_size=(494, 494), batch_size=batch_size, color_mode='grayscale', class_mode=None, shuffle=False)
        prob_left = model.predict_generator(test_generator_left)
        # print('Left: ',prob_left)
        test_generator_right = test_datagen.flow_from_directory(test_data_dir_right, target_size=(494, 494), batch_size=batch_size, color_mode='grayscale', class_mode=None, shuffle=False)
        prob_right = model.predict_generator(test_generator_right)
        # print('Right: ',prob_right)
        prob = np.maximum(prob_left, prob_right)

    print('Assessment finished. Copying files to good and bad directories....')
    # os.chdir(test_data_dir)
    os.mkdir(os.path.join('MicAssess', 'predGood'))
    os.mkdir(os.path.join('MicAssess', 'predBad'))

    good_idx = np.where(prob > args['threshold'])[0]
    bad_idx = np.where(prob <= args['threshold'])[0]

    goodlist = list(sorted(glob.glob(os.path.join('MicAssess', 'jpgs', 'data', '*.jpg')))[i] for i in good_idx)
    badlist = list(sorted(glob.glob(os.path.join('MicAssess', 'jpgs', 'data', '*.jpg')))[i] for i in bad_idx)

    pool = mp.Pool(mp.cpu_count())
    pool.map(copyGoodFile, [file for file in goodlist])
    pool.map(copyBadFile, [file for file in badlist])
    pool.close()

    # os.chdir(os.path.join(input_dir, 'MicAssess'))
    shutil.rmtree(os.path.join('MicAssess', 'jpgs')) # after prediction, remove the data directory
    if detector == 'K3':
        shutil.rmtree(os.path.join('MicAssess', 'K3Left'))
        shutil.rmtree(os.path.join('MicAssess', 'K3Right'))

    # write the output file
    # os.chdir(start_dir)
    # os.chdir(input_dir) # navigate to the par dir of input file
    try:
        os.remove(args['output'])
    except OSError:
        pass
    star_df, micname_key = utils.star2df(os.path.basename(args['input']))
    goodlist_base = [os.path.basename(f)[:-4] for f in goodlist]
    badindex = []
    for i in range(len(star_df)):
        if os.path.basename(star_df[micname_key].iloc[i])[:-4] not in goodlist_base:
            badindex.append(i)
    new_star_df = star_df.drop(badindex)
    utils.df2star(new_star_df, args['output'])
    # with open('goodlist', 'wb') as f:
    #     pickle.dump(goodlist, f)

    print('All finished!')


def main():
    args = setupParserOptions()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args['gpus']  # specify which GPU(s) to be used
    input_dir = os.path.abspath(os.path.join(args['input'], os.pardir))
    os.chdir(input_dir) # navigate to the par dir of input file/dir
    input2star(args)
    mrc2jpg(args)
    predict(args)

if __name__ == '__main__':
    main()
