#!/usr/bin/env python3
"""
Read class averages mrc file and save it to jpg.
Automatically remove the edges.
INPUT: mrcs file of 2D class averages
OUTPUT: a dir for the jpg output
The name of the jpg file would be "particlename_diamxxkxx_classnumber.jpg"
"""

import os
import mrcfile
import numpy as np
from PIL import Image
import argparse
import shutil
from .lib import imgprep
# from .lib import utils

def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help="Input mrcs file of 2D class averages.")
    ap.add_argument('-n', '--name', default='particle',
                    help="Name of the particle")
    ap.add_argument('-o', '--output', default='2DAssess',
                    help="Output jpg dir.")
    args = vars(ap.parse_args())
    return args

def mrcs2jpg(args):
    print('Converting mrcs to jpg....')
    os.chdir(os.path.abspath(os.path.dirname(args['input']))) # navigate to the par dir of input file
    try:
        shutil.rmtree(args['output'])
    except OSError:
        pass
    os.mkdir(args['output'])
    os.mkdir(os.path.join(args['output'], 'data'))

    avg_mrc = mrcfile.open(os.path.basename(args['input'])).data
    if len(avg_mrc.shape) == 3:
        num_part = avg_mrc.shape[0]
    elif len(avg_mrc.shape) == 2:
        num_part = 1

    for i in range(num_part):
        new_img = avg_mrc[i,:,:]
        if np.sum(new_img) > 1e-7 or np.sum(new_img) < -1e-7:
            new_img = imgprep.cutByRadius(new_img)
            new_img = ((new_img-new_img.min())/((new_img.max()-new_img.min())+1e-7)*255).astype('uint8')
            new_img = Image.fromarray(new_img)
            new_img = new_img.convert("L")
            new_img.save(os.path.join(args['output'], 'data', (args['name'] + '_' + str(i+1) + '.jpg')))

if __name__ == '__main__':
    args = setupParserOptions()
    mrcs2jpg(args)
