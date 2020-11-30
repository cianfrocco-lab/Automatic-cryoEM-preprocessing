#!/usr/bin/env python3
'''
Read the .mrc files, convert (and scale down with FFT) them to smaller .jpg files,
and save the .jpg files to a "data" folder under the input directory.
'''

import mrcfile
import os
import glob
import numpy as np
import argparse
import shutil
import multiprocessing as mp
import datetime
import pandas as pd
import sys
from cryoassess.lib import imgprep
from cryoassess.lib import utils

def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', help="Provide the path to the micrographs.star file.")
    ap.add_argument('-d', '--detector', default='K2', help='K2 or K3 detector?')
    ap.add_argument('--threads', type=int, default=None,
                    help='Number of threads for conversion. Dedault is None, using mp.cpu_count(). If get memory error, set it to a reasonable number.')
    args = vars(ap.parse_args())
    return args

def mrcReshape(mrc_name):
    '''
    If the shape of mrc file is 3 dims, reshape it to 2 dims (reshape with the last two).
    '''
    micrograph = mrcfile.open(mrc_name, permissive=True).data
    if len(micrograph.shape) == 3:
        micrograph = micrograph.reshape((micrograph.shape[1], micrograph.shape[2]))
        return micrograph
    else:
        return micrograph

def saveImageK2(mrc_name, height=494):
    try:
        micrograph = mrcReshape(mrc_name)
        new_img = imgprep.scaleImage(micrograph, height)

        new_img.save(os.path.join('MicAssess', 'jpgs', 'data', (os.path.basename(mrc_name)[:-4]+'.jpg')))
    except ValueError:
        print('Warning - Having trouble converting this file:', mrc_name)
        pass

def saveImageK3(mrc_name, height=494):
    try:
        micrograph = mrcReshape(mrc_name)
        new_img = imgprep.scaleImage(micrograph, height)

        short_edge = min(np.array(new_img).shape[0], np.array(new_img).shape[1])
        new_img_left = imgprep.cropLeft(np.array(new_img), short_edge, short_edge)
        new_img_right = imgprep.cropRight(np.array(new_img), short_edge, short_edge)

        new_img.save(os.path.join('MicAssess', 'jpgs', 'data', (os.path.basename(mrc_name)[:-4]+'.jpg')))
        new_img_left.save(os.path.join('MicAssess', 'K3Left', 'data', (os.path.basename(mrc_name)[:-4]+'.jpg')))
        new_img_right.save(os.path.join('MicAssess', 'K3Right', 'data', (os.path.basename(mrc_name)[:-4]+'.jpg')))
    except ValueError:
        print('Warning - Having trouble converting this file:', mrc_name)
        pass

def mrc2jpg(args):
    # input_dir = os.path.abspath(os.path.join(args['input'], os.pardir))
    # os.chdir(input_dir) # navigate to the par dir of input file
    # os.chdir(os.path.abspath(os.path.dirname(args['input'])) # navigate to the par dir of input file
    mic_list = utils.star2miclist(os.path.basename(args['input']))
    try:
        shutil.rmtree('MicAssess')
    except OSError:
        pass
    os.mkdir('MicAssess')
    os.mkdir(os.path.join('MicAssess', 'jpgs'))
    os.mkdir(os.path.join('MicAssess', 'jpgs', 'data'))

    if args['detector'] == 'K3':
        os.mkdir(os.path.join('MicAssess', 'K3Left'))
        os.mkdir(os.path.join('MicAssess', 'K3Right'))
        os.mkdir(os.path.join('MicAssess', 'K3Left', 'data'))
        os.mkdir(os.path.join('MicAssess', 'K3Right', 'data'))

    if args['threads'] == None:
        num_threads = mp.cpu_count()
    else:
        num_threads = args['threads']
    # pool = mp.Pool(mp.cpu_count())
    # print('CPU count is ', mp.cpu_count())
    pool = mp.Pool(num_threads)
    print('Thread count is ', num_threads)
    if args['detector'] == 'K2':
        pool.map(saveImageK2, [mrc_name for mrc_name in mic_list])
        pool.close()
    elif args['detector'] == 'K3':
        pool.map(saveImageK3, [mrc_name for mrc_name in mic_list])
        pool.close()
    print('Conversion finished.')

if __name__ == '__main__':
    args = setupParserOptions()
    mrc2jpg(args)
