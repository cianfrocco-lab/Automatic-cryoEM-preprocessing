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
from PIL import Image
from PIL import ImageOps
import shutil
import multiprocessing as mp
import datetime
import pandas as pd
import sys

def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help="Provide the path to the micrographs.star file.")
    # ap.add_argument('-w', '--width', default='512',
    #                 help="The width of the .jpg image after scaling down. Hardcoded to 512 for now, not functional.")
    args = vars(ap.parse_args())
    return args

def star2miclist(starfile):
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
    mic_list = star_df['_rlnMicrographName\n'].tolist()

    return mic_list

def downsample(x, width=512):
    """ Downsample 2d array using fourier transform """
    m,n = x.shape[-2:]
    # factor = width/n
    factor = n/width
    F = np.fft.rfft2(x)
    #F = np.fft.fftshift(F)
    S = round(2*factor)
    A = F[...,0:m//S,0:n//S+2]
    B = F[...,-m//S+1:,0:n//S+2]
    F = np.concatenate([A,B], axis=-2)
    f = np.fft.irfft2(F)
    return f

def scale_image(img, width=512):
    new_img = downsample(img, width)
    new_img = ((new_img-new_img.min())/((new_img.max()-new_img.min())+1e-7)*255).astype('uint8')
    new_img = Image.fromarray(new_img)
    new_img = new_img.convert("L")
    return new_img

def save_image(mrc_name, width=512):
    try:
        micrograph = mrcfile.open(mrc_name, permissive=True).data
        new_img = scale_image(micrograph, width)
        new_img.save(os.path.join('MicAssess', 'data', (os.path.basename(mrc_name)[:-4]+'.jpg')))
    except ValueError:
        print('Warning - Having trouble converting this file:', mrc_name)
        pass

def mrc2jpg(**args):
    os.chdir(os.path.abspath(os.path.dirname(args['input']))) # navigate to the par dir of input file
    mic_list = star2miclist(os.path.basename(args['input']))
    try:
        shutil.rmtree('MicAssess')
    except OSError:
        pass
    os.mkdir('MicAssess')
    os.mkdir(os.path.join('MicAssess', 'data'))

    pool = mp.Pool(mp.cpu_count())
    print('CPU count is ', mp.cpu_count())
    pool.map(save_image, [mrc_name for mrc_name in mic_list])
    pool.close()
    print('Conversion finished.')

if __name__ == '__main__':
    args = setupParserOptions()
    mrc2jpg(**args)
