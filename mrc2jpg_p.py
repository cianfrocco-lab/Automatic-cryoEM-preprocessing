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
    ap.add_argument('-i', '--input', help="Provide the path to the micrographs.star file.")
    ap.add_argument('-d', '--detector', default='K2', help='K2 or K3 detector?')
    ap.add_argument('--threads', type=int, default=None,
                    help='Number of threads for conversion. Dedault is None, using mp.cpu_count(). If get memory error, set it to a reasonable number.')
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
    micname_key = [x for x in keys if 'MicrographName' in x][0]
    mic_list = star_df[micname_key].tolist()

    return mic_list

def downsample(x, height=494):
    """ Downsample 2d array using fourier transform """
    m,n = x.shape[-2:]
    # factor = width/n
    factor = m/height
    width = round(n/factor/2)*2
    F = np.fft.rfft2(x)
    A = F[...,0:height//2,0:width//2+1]
    B = F[...,-height//2:,0:width//2+1]
    F = np.concatenate([A,B], axis=0)
    # S = round(2*factor)
    # A = F[...,0:m//S,0:n//S+2]
    # B = F[...,-m//S+1:,0:n//S+2]
    # F = np.concatenate([A,B], axis=-2)
    f = np.fft.irfft2(F, s=(height, width))
    return f

def scale_image(img, height=494):
    new_img = downsample(img, height)
    new_img = ((new_img-new_img.min())/((new_img.max()-new_img.min())+1e-7)*255).astype('uint8')
    new_img = Image.fromarray(new_img)
    new_img = new_img.convert("L")
    return new_img

def crop_left(img,cropx,cropy):
    y = img.shape[0]
    startx = 0
    starty = y//2-(cropy//2)
    new_img_left = img[starty:starty+cropy,startx:startx+cropx]
    new_img_left = Image.fromarray(new_img_left)
    new_img_left = new_img_left.convert("L")
    return new_img_left

def crop_right(img,cropx,cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = x-cropx
    starty = y//2-(cropy//2)
    new_img_right = img[starty:starty+cropy,startx:startx+cropx]
    new_img_right = Image.fromarray(new_img_right)
    new_img_right = new_img_right.convert("L")
    return new_img_right

def save_image_k2(mrc_name, height=494):
    try:
        micrograph = mrcfile.open(mrc_name, permissive=True).data
        if len(micrograph.shape) == 3:
            micrograph = micrograph.reshape((micrograph.shape[1], micrograph.shape[2]))
        else:
            micrograph = micrograph
        new_img = scale_image(micrograph, height)
        new_img.save(os.path.join('MicAssess', 'jpgs', 'data', (os.path.basename(mrc_name)[:-4]+'.jpg')))
    except ValueError:
        print('Warning - Having trouble converting this file:', mrc_name)
        pass

def save_image_k3(mrc_name, height=494):
    try:
        micrograph = mrcfile.open(mrc_name, permissive=True).data
        if len(micrograph.shape) == 3:
            micrograph = micrograph.reshape((micrograph.shape[1], micrograph.shape[2]))
        else:
            micrograph = micrograph
        new_img = scale_image(micrograph, height)
        short_edge = min(np.array(new_img).shape[0], np.array(new_img).shape[1])
        new_img_left = crop_left(np.array(new_img), short_edge, short_edge)
        new_img_right = crop_right(np.array(new_img), short_edge, short_edge)
        new_img.save(os.path.join('MicAssess', 'jpgs', 'data', (os.path.basename(mrc_name)[:-4]+'.jpg')))
        new_img_left.save(os.path.join('MicAssess', 'k3_left', 'data', (os.path.basename(mrc_name)[:-4]+'.jpg')))
        new_img_right.save(os.path.join('MicAssess', 'k3_right', 'data', (os.path.basename(mrc_name)[:-4]+'.jpg')))
    except ValueError:
        print('Warning - Having trouble converting this file:', mrc_name)
        pass

def mrc2jpg(**args):
    os.chdir(os.path.dirname(args['input'])) # navigate to the par dir of input file
    # os.chdir(os.path.abspath(os.path.dirname(args['input'])) # navigate to the par dir of input file
    mic_list = star2miclist(os.path.basename(args['input']))
    try:
        shutil.rmtree('MicAssess')
    except OSError:
        pass
    os.mkdir('MicAssess')
    os.mkdir(os.path.join('MicAssess', 'jpgs'))
    os.mkdir(os.path.join('MicAssess', 'jpgs', 'data'))

    if args['detector'] == 'K3':
        os.mkdir(os.path.join('MicAssess', 'k3_left'))
        os.mkdir(os.path.join('MicAssess', 'k3_right'))
        os.mkdir(os.path.join('MicAssess', 'k3_left', 'data'))
        os.mkdir(os.path.join('MicAssess', 'k3_right', 'data'))

    if args['threads'] == None:
        num_threads = mp.cpu_count()
    else:
        num_threads = args['threads']
    # pool = mp.Pool(mp.cpu_count())
    # print('CPU count is ', mp.cpu_count())
    pool = mp.Pool(num_threads)
    print('Thread count is ', num_threads)
    if args['detector'] == 'K2':
        pool.map(save_image_k2, [mrc_name for mrc_name in mic_list])
        pool.close()
    elif args['detector'] == 'K3':
        pool.map(save_image_k3, [mrc_name for mrc_name in mic_list])
        pool.close()
    print('Conversion finished.')

if __name__ == '__main__':
    args = setupParserOptions()
    mrc2jpg(**args)
