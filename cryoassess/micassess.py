import os
import numpy as np
import glob
import shutil
import multiprocessing as mp
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from cryoassess.lib import utils, fft, star, mrc2png


IMG_DIM = 494
LABEL_LIST = ['0Great', '1Decent', '2Contamination_Aggregate_Crack_Breaking_Drifting', '3Empty_no_ice', '4Crystalline_ice', '5Empty_ice_no_particles_but_vitreous_ice']


def setupParserOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',
                    help="Input directory, starfile with a list of micrographs or a pattern "
                         "(<path to a folder>/<pattern>. Pattern could have *, ?, any valid glob wildcard.  "
                         "All of the micrographs must be in mrc format. Cannot contain other directories inside (excluding directories made by MicAssess).",
                    required=True)
    ap.add_argument('-d', '--detector', default='K2', help='K2 or K3 detector?')
    ap.add_argument('-m', '--model', default='./models', help='Path to all the model files.')
    ap.add_argument('-o', '--output', default='MicAssess', help="Name of the output directory. Default is MicAssess.")
    ap.add_argument('-b', '--batch_size', type=int, default=32,
                    help="Batch size used in prediction. Default is 32. If memory error/warning appears, try lower this number to 16, 8, or even lower.")
    ap.add_argument('--t1', type=float, default=0.1,
                    help="Threshold for good/bad classification. Default is 0.1. Higher number will cause more good micrographs (including great and good) being classified as bad. On the other hand, if you find good micrographs misclassified as bad, try to lower this number.")
    ap.add_argument('--t2', type=float, default=0.1,
                    help="Threshold for great/decent classification. Default is 0.1. Higher number will cause more great micrographs being classified as good.")
    ap.add_argument('--threads', type=int, default=None,
                    help='Number of threads for conversion. Default is None, using mp.cpu_count(). If get memory error, set it to a reasonable number.')
    ap.add_argument('--gpus', default='0', help='Specify which gpu(s) to use, e.g. 0,1. Default is 0, which uses only one gpu.')
    ap.add_argument('--dont_reset', default=False, action='store_true',
                    help='If you already have the mrc files converted (to png) with a previous run of MicAssess, you can skip the conversion step by using this flag.')
    args = vars(ap.parse_args())
    return args


def reset():
    try:
        shutil.rmtree('MicAssess')
    except OSError:
        pass

def dont_reset():
    try:
        for name in LABEL_LIST:
            shutil.rmtree(os.path.join('MicAssess', name))
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

    if args['detector'] == 'K2':
        img_w = 512
    elif args['detector'] == 'K3':
        img_w = 696

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMG_DIM, img_w),
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


def build_models(args, cutpos):

    base_model_path = glob.glob(os.path.join(args['model'], 'base_*'))[0]
    binary_head_path = glob.glob(os.path.join(args['model'], 'fine_binary_*'))[0]
    good_head_path = glob.glob(os.path.join(args['model'], 'fine_good_*'))[0]
    bad_head_path = glob.glob(os.path.join(args['model'], 'fine_bad_*'))[0]

    if args['detector'] == 'K2':
        img_w = 512
    elif args['detector'] == 'K3':
        img_w = 696

    inputs = Input(shape=(IMG_DIM, img_w, 1))

    if args['detector'] == 'K2' and cutpos == 'center':
        crop = Cropping2D(cropping=((0, 0), (9, 9)))(inputs)
    elif args['detector'] == 'K3' and cutpos == 'left':
        crop = Cropping2D(cropping=((0, 0), (0, 202)))(inputs)
    elif args['detector'] == 'K3' and cutpos == 'right':
        crop = Cropping2D(cropping=((0, 0), (202, 0)))(inputs)

    base_model = load_model(base_model_path)
    base_model = Model(base_model.inputs, base_model.layers[-2].output)

    r_features = base_model(crop, training=False)
    f_features = Lambda(fft.radavg_logps_sigmoid_tf, name='f_features')(crop)
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

    return base_model, binary_head, good_head, bad_head


def predict(args):

    print('Assessing micrographs....')

    test_data_dir = os.path.join(args['output'], 'png')

    strategy = tf.distribute.MirroredStrategy(devices=None)
    print('[INFO]: Number of devices: {}'.format(strategy.num_replicas_in_sync))

    if args['detector'] == 'K2':
        with strategy.scope():
            base_model, binary_head, good_head, bad_head = build_models(args, cutpos='center')
            test_datagen = ImageDataGenerator(preprocessing_function=utils.preprocess_c)
            probs, fine_good_probs, fine_bad_probs = predict_one(test_datagen, test_data_dir, base_model, binary_head, good_head, bad_head, args)

    elif args['detector'] == 'K3':
        with strategy.scope():
            base_model, binary_head, good_head, bad_head = build_models(args, cutpos='left')
            test_datagen = ImageDataGenerator(preprocessing_function=utils.preprocess_l)
            probs_l, fine_good_probs_l, fine_bad_probs_l = predict_one(test_datagen, test_data_dir, base_model, binary_head, good_head, bad_head, args)

            base_model, binary_head, good_head, bad_head = build_models(args, cutpos='right')
            test_datagen = ImageDataGenerator(preprocessing_function=utils.preprocess_r)
            probs_r, fine_good_probs_r, fine_bad_probs_r = predict_one(test_datagen, test_data_dir, base_model, binary_head, good_head, bad_head, args)

        probs = np.minimum(probs_l, probs_r)
        fine_good_probs = np.mean([fine_good_probs_l, fine_good_probs_r], axis=0)
        fine_bad_probs = np.mean([fine_bad_probs_l, fine_bad_probs_r], axis=0)

    probsFile_1 = 'probs_good.tsv'
    with open(probsFile_1, 'w') as f:
        for i in range(len(probs)):
            micName = sorted(glob.glob(os.path.join(args['output'], 'png', 'data', '*.png')))[i]
            f.write('{}\t{}\n'.format(micName.split('/')[-1], (1 - probs[i][0])))

    probsFile_2 = 'probs_great.tsv'
    with open(probsFile_2, 'w') as f:
        for i in range(len(fine_good_probs)):
            micName = sorted(glob.glob(os.path.join(args['output'], 'png', 'data', '*.png')))[i]
            f.write('{}\t{}\n'.format(micName.split('/')[-1], (1 - fine_good_probs[i][0])))

    return probs, fine_good_probs, fine_bad_probs


def assign_label(prob, fine_good_prob, fine_bad_prob, threshold_1, threshold_2):
    if prob[0] <= threshold_1: # meaning this is a good mic
        if fine_good_prob[0] < threshold_2:
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
        idx = np.where(labels==i)[0]
        if idx.size:
            if i == 0:
                greatlist = list(sorted(glob.glob(os.path.join(test_data_dir, 'data', '*.png')))[ii] for ii in idx)
            if i == 1:
                goodlist = list(sorted(glob.glob(os.path.join(test_data_dir, 'data', '*.png')))[ii] for ii in idx)
            for j in idx:
                file = sorted(glob.glob(os.path.join(test_data_dir, 'data', '*.png')))[int(j)]
                shutil.copy2(file, os.path.join(args['output'], LABEL_LIST[i]))

    goodlist = goodlist + greatlist
    # shutil.rmtree(test_data_dir)

    return goodlist, greatlist


def write_star(args, goodlist, greatlist):

    try:
        os.remove(os.path.join(os.path.dirname(args['input']), os.path.splitext(os.path.basename(args['input']))[0] + '_great.star'))
    except OSError:
        pass

    try:
        os.remove(os.path.join(os.path.dirname(args['input']), os.path.splitext(os.path.basename(args['input']))[0] + '_good.star'))
    except OSError:
        pass

    if greatlist:
        star_df = star.star2df(args['input'])
        mic_blockcode = star.micBlockcode(star_df)
        greatlist_base = [os.path.basename(f)[:-4] for f in greatlist]
        omitindex2 = []
        for i in range(len(star_df[mic_blockcode][0])):
            if os.path.basename(star_df[mic_blockcode][0]['_rlnMicrographName'].iloc[i])[:-4] not in greatlist_base:
                omitindex2.append(i)
        star_df[mic_blockcode][0].drop(omitindex2, inplace=True)
        star.df2star(star_df, os.path.join(os.path.dirname(args['input']), os.path.splitext(os.path.basename(args['input']))[0] + '_great.star'))
    else:
        print('No "great" micrographs found.')

    if goodlist:
        star_df = star.star2df(args['input'])
        mic_blockcode = star.micBlockcode(star_df)
        goodlist_base = [os.path.basename(f)[:-4] for f in goodlist]
        omitindex1 = []
        for i in range(len(star_df[mic_blockcode][0])):
            if os.path.basename(star_df[mic_blockcode][0]['_rlnMicrographName'].iloc[i])[:-4] not in goodlist_base:
                omitindex1.append(i)
        star_df[mic_blockcode][0].drop(omitindex1, inplace=True)
        star.df2star(star_df, os.path.join(os.path.dirname(args['input']), os.path.splitext(os.path.basename(args['input']))[0] + '_good.star'))
    else:
        print('No "good" micrographs found.')


def report(labels, greatlist, goodlist):
    print('Total:\t %d micrographs' %len(labels))
    for i in range(len(LABEL_LIST)):
        print(LABEL_LIST[i], ':\t %d micrographs' %(len(np.where(labels==i)[0])))

    perc_great = "{:.2f}".format(100 * len(greatlist) / len(labels))
    perc_good = "{:.2f}".format(100 * len(goodlist) / len(labels))
    print('%s%% of the micrographs are great and were written in the *_great.star file.' %perc_great)
    print('%s%% of the micrographs are good and were written in the *_good.star file.' %perc_good)
    print('Details can be found in the output directory.')



def main():
    args = setupParserOptions()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args['gpus']  # specify which GPU(s) to be used

    if not args['dont_reset']:
        reset()
        input2star(args)
        mrc2png.mrc2png(args)
    else:
        dont_reset()
        print('Skipping the conversion step.')

    probs, fine_good_probs, fine_bad_probs = predict(args)
    # print(probs, fine_good_probs, fine_bad_probs)
    labels = assign_labels(probs, fine_good_probs, fine_bad_probs, (1-args['t1']), (1-args['t2']))
    goodlist, greatlist = loop_files(labels, args)
    write_star(args, goodlist, greatlist)
    report(labels, greatlist, goodlist)


if __name__ == '__main__':
    main()
