import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def normalize(x):
    x /= 127.5
    x -= 1.
    return x


def fft2(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))

def fft2_tf(x):
    x = tf.cast(x, tf.complex64)
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(x)))

def power_spectrum(x):
    return np.abs(fft2(x))**2

def power_spectrum_tf(x):
    return tf.math.pow(tf.math.abs(fft2_tf(x)), 2)


def radial_avg(img):

    x0 = img.shape[1] // 2
    y0 = img.shape[0] // 2

    x,y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    R = np.sqrt((x-x0)**2 + (y-y0)**2)

    # calculate the mean
    f = lambda r : img[(R >= r-.5) & (R < r+.5)].mean()
    img_r = min(img.shape[1], img.shape[0]) // 2
    r = np.linspace(1, img_r, num=img_r)
    mean = np.vectorize(f)(r)

    return r, mean

def radial_avg_tf(img):

    x0 = tf.math.divide(img.shape[2], 2)
    x0 = tf.cast(x0, tf.float32)
    y0 = tf.math.divide(img.shape[1], 2)
    y0 = tf.cast(y0, tf.float32)

    x,y = tf.meshgrid(tf.range(img.shape[2]), tf.range(img.shape[1]))
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    R = tf.math.sqrt(tf.math.square(x-x0) + tf.math.square(y-y0))
    # R = K.tile(K.expand_dims(R, axis=0),[tf.shape(img)[0], 1, 1])

    # calculate the mean
    # f = lambda r : tf.reduce_mean(img[(R >= r-.5) & (R < r+.5)])
    f = lambda r : tf.reduce_mean(tf.boolean_mask(img, (R >= r-.5) & (R < r+.5), axis=1), axis=1)

    img_r = tf.math.floordiv(tf.math.minimum(img.shape[2], img.shape[1]), 2)
    img_r = tf.cast(img_r, tf.float32)
    r = tf.linspace(1.0, img_r, num=tf.cast(img_r, tf.int32))
    mean = tf.map_fn(f, r)

    mean = tf.transpose(tf.squeeze(mean))

    return r, mean


def radavg_logps(x, normalize=False):

    if normalize:
        x = normalize(x)
    ps_x = power_spectrum(x)
    r, mean = radial_avg(np.log(ps_x))

    return mean

def radavg_logps_tf(x):
    '''
    No normalization.
    '''
    ps_x = power_spectrum_tf(x)
    r, mean = radial_avg_tf(tf.math.log(ps_x))
    # mean = K.expand_dims(mean, axis=0)

    return mean

def radavg_logps_sigmoid_tf(x):
    return tf.math.sigmoid(radavg_logps_tf(x))



# import cv2
# import matplotlib.pyplot as plt
# x = cv2.imread(r'C:\Users\Mutania\Desktop\micrograph.png', cv2.IMREAD_GRAYSCALE).astype('float64')
# x = normalize(x)
# fft_x = fft2(x)
# ps_x = power_spectrum(x)
# plt.imshow(np.log(ps_x), cmap='gray')

# r, mean = radial_avg(np.log(ps_x))
# plt.plot(r, mean)

# x = tf.convert_to_tensor(x)
# mean = radavg_logps_tf(x)



# import cv2, os
# import utils

# allimg_path = r'C:\Users\Mutania\Desktop\MicAssess_v1.0_data\micassess_train_k2\data'
# all_mean = []

# i = 0
# for f in os.listdir(allimg_path):
#     fname = os.path.join(allimg_path, f)
#     x = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype('float64')
#     x = utils.preprocess(x)
#     mean = radavg_logps(x, normalize=False)
#     all_mean.append(mean)
#     i += 1
#     if i % 500 == 0:
#         print(i)
#     if i == 2000:
#         break

# avg_mean = np.mean(all_mean, axis=0)
# std_mean = np.std(all_mean, axis=0)
# np.save('radavg_mean.npy', avg_mean)
# np.save('radavg_std.npy', std_mean)

# max_mean = np.max(all_mean, axis=0)
# min_mean = np.min(all_mean, axis=0)
# np.save('radavg_max.npy', max_mean)
# np.save('radavg_min.npy', min_mean)
