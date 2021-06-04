#!/usr/bin/env python3
'''
Read mrc files, use FFT to downsample them to smaller png files.
The output png files will have height as 494 px and h/w ratio will be kept.
'''

import mrcfile
import os
import glob
import numpy as np
import argparse
from PIL import Image
import multiprocessing as mp
from pathlib import Path
import star

def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', help="Provide the path to the micrographs.star file.")
    ap.add_argument('-o', '--output', help='Provide the path to the output directory.')
    ap.add_argument('--threads', type=int, default=None,
                    help='Number of threads for conversion. Default is None, using mp.cpu_count(). If get memory error, set it to a reasonable number.')
    args = vars(ap.parse_args())
    return args

def downsample(img, height=494):
    '''
    Downsample 2d array using fourier transform.
    factor is the downsample factor.
    '''
    m,n = img.shape[-2:]
    ds_factor = m/height
    width = round(n/ds_factor/2)*2
    F = np.fft.rfft2(img)
    A = F[...,0:height//2,0:width//2+1]
    B = F[...,-height//2:,0:width//2+1]
    F = np.concatenate([A, B], axis=0)
    f = np.fft.irfft2(F, s=(height, width))
    return f

def scale_image(img, height=494):
    newImg = downsample(img, height)
    newImg = ((newImg - newImg.min()) / ((newImg.max() - newImg.min()) + 1e-7) * 255)
    newImg = Image.fromarray(newImg).convert('L')
    return newImg

def save_image(mrc_name, outdir, height=494):
    # print(mrc_name)
    try:
        micrograph = mrcfile.open(mrc_name, permissive=True).data
        micrograph = micrograph.reshape((micrograph.shape[-2], micrograph.shape[-1]))
        newImg = scale_image(micrograph, height)
        newImg.save(os.path.join(outdir, os.path.splitext(os.path.basename(mrc_name))[0] + '.png'))
    except ValueError:
        print('An error occured when trying to save ', mrc_name)
        pass

def mrc2png(args):
    # os.chdir(args['output'])
    mic_list = star.star2miclist(args['input'])
    Path(os.path.join(args['output'], 'png', 'data')).mkdir(parents=True, exist_ok=True)
    threads = mp.cpu_count() if args['threads'] is None else args['threads']
    with mp.Pool(threads) as pool:
        print('Converting in %d parallel threads....' %threads)
        pool.starmap(save_image, ((mrc_name, os.path.join(args['output'], 'png', 'data')) for mrc_name in mic_list))
    print('Conversion finished.')

if __name__ == '__main__':
    args = setupParserOptions()
    mrc2png(args)
