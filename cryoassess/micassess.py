import os
import numpy as np
import glob
import shutil
import multiprocessing as mp
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from lib import utils, fft, star, mrc2png


IMG_DIM = 494
# BATCH_SIZE = 32
# threshold_1 = 0.5
# threshold_2 = 0.5

LABEL_LIST = ['0Great', '1Good', '2Contamination_Aggregate_Crack_Breaking_drifting', '3Empty_no_ice', '4Crystalline_ice', '5Empty_ice_no_particles_but_vitreous_ice']

# test_data_dir = '/lsi/groups/mcianfroccolab/yilai/MicAssess_v1.0_test/MicAssess/png'

# base_model_path = '/lsi/groups/mcianfroccolab/yilai/codes/cryoassess-train/models/base_resnext50_05212021_lr1e-3_b32.h5'
# binary_head_path = '/lsi/groups/mcianfroccolab/yilai/codes/cryoassess-train/models/fine_binary_resnext50_ps_head_05252021_lr5e-6_b32.h5'
# good_head_path = '/lsi/groups/mcianfroccolab/yilai/codes/cryoassess-train/models/fine_good_resnext50_ps_head_05252021_lr1e-6_b16.h5'
# bad_head_path = '/lsi/groups/mcianfroccolab/yilai/codes/cryoassess-train/models/fine_bad_resnext50_ps_head_05252021_lr5e-6_b16.h5'



def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help="Input directory, starfile with a list of micrographs or a pattern "
                         "(<path to a folder>/<pattern>. Pattern could have *, ?, any valid glob wildcard.  "
                         "All of the micrographs must be in mrc format. Cannot contain other directories inside (excluding directories made by MicAssess).",
                    required=True)
    ap.add_argument('-d', '--detector', default='K2', help='K2 or K3 detector?')
    ap.add_argument('-m', '--model', default='/lsi/groups/mcianfroccolab/yilai/codes/cryoassess-train/models', help='Path to all the model files.')
    ap.add_argument('-o', '--output', default='MicAssess', help="Name of the output directory. Default is MicAssess.")
    ap.add_argument('-b', '--batch_size', type=int, default=32,
                    help="Batch size used in prediction. Default is 32. If memory error/warning appears, try lower this number to 16, 8, or even lower.")
    ap.add_argument('--t1', type=float, default=0.2,
                    help="Threshold for good/bad classification. Default is 0.2. Higher number will cause more good micrographs (including great and good) being classified as bad.")
    ap.add_argument('--t2', type=float, default=0.5,
                    help="Threshold for great/good classification. Default is 0.5. Higher number will cause more great micrographs being classified as good.")
    ap.add_argument('--threads', type=int, default=None,
                    help='Number of threads for conversion. Default is None, using mp.cpu_count(). If get memory error, set it to a reasonable number.')
    ap.add_argument('--not_reset', default=False, action='store_true',
                    help='Do not reset and clear the existing job (if any). Turning this off will skip the converting to png if the output directory already exists.')
    ap.add_argument('--gpus', default='0', help='Specify which gpu(s) to use, e.g. 0,1. Default is 0, which uses only one gpu.')
    args = vars(ap.parse_args())
    return args



def reset():
    try:
        shutil.rmtree('MicAssess')
    except OSError:
        pass



def input2star(args):
    input = args['input']
    # if a star file
    if input.endswith('.star'):
        return

    micList = []
    # input = os.path.basename(input)
    micList = glob.glob(input)

    # Get the dirname
    # folder = os.path.dirname(input)
    # newStarFile = os.path.join(folder, "micA_micrographs.star")
    newStarFile = "micrographs.star"
    print("Generating star file %s" % newStarFile)
    if os.path.exists(newStarFile):
        print("Previous star file found, overwriting.")
        os.remove(newStarFile)
    f = open(newStarFile, "w")
    f.write("data_\n")
    f.write("loop_\n")
    f.write("_rlnMicrographName\n")
    f.writelines('\n'.join(micList))
    f.close()

    args['input'] = newStarFile




def predict_one(test_datagen, test_data_dir, base_model, binary_head, good_head, bad_head, args):

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMG_DIM, IMG_DIM),
        batch_size=args['batch_size'],
        color_mode='grayscale',
        class_mode=None,
        shuffle=False,
        )

    features = base_model.predict(test_generator)

    probs = binary_head.predict(features)
    fine_good_probs = good_head.predict(features)
    fine_bad_probs = bad_head.predict(features)

    return probs, fine_good_probs, fine_bad_probs




def predict(args):

    base_model_path = glob.glob(os.path.join(args['model'], 'base_*'))[0]
    binary_head_path = glob.glob(os.path.join(args['model'], 'fine_binary_*'))[0]
    good_head_path = glob.glob(os.path.join(args['model'], 'fine_good_*'))[0]
    bad_head_path = glob.glob(os.path.join(args['model'], 'fine_bad_*'))[0]
    detector = args['detector']
    test_data_dir = os.path.join(args['output'], 'png')

    print('Assessing micrographs....')

    base_model = load_model(base_model_path)
    base_model = Model(base_model.inputs, base_model.layers[-2].output)
    inputs = Input(shape=(IMG_DIM, IMG_DIM, 1))
    r_features = base_model(inputs, training=False)
    f_features = Lambda(fft.radavg_logps_sigmoid_tf, name='f_features')(inputs)
    f_features = tf.reshape(f_features, (tf.shape(inputs)[0], 247))
    features = Concatenate(axis=1)([r_features, f_features])
    base_model = Model(inputs, features)

    binary_head = load_model(binary_head_path)
    binary_head.trainable = False
    binary_head.compile(optimizer = Adam(learning_rate = 5e-6), loss = 'binary_crossentropy', metrics = ['accuracy'])

    good_head = load_model(good_head_path)
    good_head.trainable = False
    good_head.compile(optimizer = Adam(learning_rate = 5e-6), loss = 'binary_crossentropy', metrics = ['accuracy'])

    bad_head = load_model(bad_head_path)
    bad_head.trainable = False
    bad_head.compile(optimizer=Adam(learning_rate = 5e-6), loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])

    if detector == 'K2':
        test_datagen = ImageDataGenerator(preprocessing_function=utils.preprocess_c)
        probs, fine_good_probs, fine_bad_prosb = predict_one(test_datagen, test_data_dir, base_model, binary_head, good_head, bad_head, args)

    elif detector == 'K3':
        test_datagen = ImageDataGenerator(preprocessing_function=utils.preprocess_l)
        probs_l, fine_good_probs_l, fine_bad_probs_l = predict_one(test_datagen, test_data_dir, base_model, binary_head, good_head, bad_head, args)

        test_datagen = ImageDataGenerator(preprocessing_function=utils.preprocess_r)
        probs_r, fine_good_probs_r, fine_bad_probs_r = predict_one(test_datagen, test_data_dir, base_model, binary_head, good_head, bad_head, args)

        probs = np.maximum(probs_l, probs_r)
        fine_good_probs = np.mean([fine_good_probs_l, fine_good_probs_r], axis=0)
        fine_bad_probs = np.mean([fine_bad_probs_l, fine_bad_probs_r], axis=0)

    return probs, fine_good_probs, fine_bad_probs



def assign_label(prob, fine_good_prob, fine_bad_prob, threshold_1, threshold_2):
    if prob[0] <= threshold_1: # meaning this is a good mic
        if fine_good_prob[0] > threshold_2:
            label = 0 # 0Great
        else:
            label = 1 # 1Good
    else: # this is a bad mic
        label = np.argmax(fine_bad_prob) + 2
    return label


def assign_labels(probs, fine_good_probs, fine_bad_probs, threshold_1, threshold_2):
    labels = np.full(shape=probs.shape[0], fill_value=99)
    for i in range(probs.shape[0]):
        labels[i] = assign_label(probs[i], fine_good_probs[i], fine_bad_probs[i], threshold_1, threshold_2)
    return labels




def loop_files(labels, args):

    test_data_dir = os.path.join(args['output'], 'png')
    for label_name in LABEL_LIST:
        Path(os.path.join(args['output'], label_name)).mkdir(parents=True, exist_ok=True)

    goodlist = []
    greatlist = []
    for i in range(len(LABEL_LIST)):
        idx = np.where(labels==i)
        if i == 0:
            greatlist = list(sorted(glob.glob(os.path.join(test_data_dir, 'data', '*.png'))) for ii in idx)
        if i < 2:
            goodlist = list(sorted(glob.glob(os.path.join(test_data_dir, 'data', '*.png'))) for ii in idx)
        for j in idx:
            file = sorted(glob.glob(os.path.join(test_data_dir, 'data', '*.png')))[j]
            shutil.copy2(file, os.path.join(args['output'], label_list[i]))

    shutil.rmtree(test_data_dir)

    return goodlist, greatlist




def write_star(args, goodlist, greatlist):

    star_df = star.star2df(args['input'])
    mic_blockcode = star.micBlockcode(star_df)

    goodlist_base = [os.path.basename(f)[:-4] for f in goodlist]
    omitindex1 = []
    for i in range(len(star_df[mic_blockcode][0])):
        if os.path.basename(star_df[mic_blockcode][0]['_rlnMicrographName'].iloc[i])[:-4] not in goodlist_base:
            omitindex1.append(i)
    star_df[mic_blockcode][0].drop(omitindex1, inplace=True)
    utils.df2star(star_df, os.path.join(os.path.dirname(args['input']), os.path.splitext(os.path.basename(args['input']))[0] + '_good.star'))

    greatlist_base = [os.path.basename(f)[:-4] for f in goodlist]
    omitindex2 = []
    for i in range(len(star_df[mic_blockcode][0])):
        if os.path.basename(star_df[mic_blockcode][0]['_rlnMicrographName'].iloc[i])[:-4] not in greatlist_base:
            omitindex2.append(i)
    star_df[mic_blockcode][0].drop(omitindex2, inplace=True)
    utils.df2star(star_df, os.path.join(os.path.dirname(args['input']), os.path.splitext(os.path.basename(args['input']))[0] + '_great.star'))





def main():
    args = setupParserOptions()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args['gpus']  # specify which GPU(s) to be used

    if not args['not_reset']:
        print('Resetting....')
        reset()
        print('Done')

    input2star(args)
    mrc2png.mrc2png(args)
    probs, fine_good_probs, fine_bad_probs = predict(args)
    labels = assign_labels(probs, fine_good_probs, fine_bad_probs, args['t1'], args['t2'])
    goodlist, greatlist = loop_files(labels, args)
    write_star(args, goodlist, greatlist)









if __name__ == '__main__':
    main()
