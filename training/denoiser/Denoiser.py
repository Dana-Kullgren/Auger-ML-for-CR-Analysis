import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import backend as K
##########################################################################################################################################
## Include this for the ouput file to look cleaner
import logging
import os
def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)
# set_tf_loglevel(logging.INFO)
set_tf_loglevel(logging.FATAL)
##########################################################################################################################################
# Custom Functions
from tensorflow.keras import backend as K
def CorrCoeff(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = mx
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    return r

# # custom correlation metric
# def CC2(y_true, y_pred):
#     def pearson_correlation(x, y):
#         mean_x = K.mean(x)
#         mean_y = K.mean(y)
#         cov_xy = K.sum((x - mean_x) * (y - mean_y))
#         std_x = K.sqrt(K.sum(K.square(x - mean_x)))
#         std_y = K.sqrt(K.sum(K.square(y - mean_y)))
#         if std_x == 0 or std_y == 0:
#             print(f'std_x = {std_x}, std_y = {std_y}')
#         correlation = cov_xy / (std_x * std_y)
#         return correlation

#     correlation = tf.py_function(pearson_correlation, [y_true, y_pred], tf.float32)
#     return correlation

# def ShiftTraces(Sig, Noisy):
#     # Takes all Sig and Noisy traces and roll them (to move the peak locations)
#     import random
#     shiftedSig = []
#     shiftedNoisy = []
#     for i in range(len(Sig)):
#         shifInd = random.randint(-1500,1500) # shifting by random no of bins
#         shif =  np.roll(Sig[i], shifInd)
#         shifN = np.roll(Noisy[i], shifInd)

#         shiftedSig.append(shif)
#         shiftedNoisy.append(shifN)

#     shiftedSig = np.array(shiftedSig)
#     shiftedNoisy = np.array(shiftedNoisy)
#     return shiftedSig, shiftedNoisy

def Normalizing(signal, noised_signal): ## Normalize by the peak of Noisy traces.
    Signals = np.asarray(signal)
    NoisySig = np.asarray(noised_signal)
    # Normalization
    signals_shift = []
    Noisy_shift = []
    mapping = []
    for i, j in zip(Signals, NoisySig):
        j_m = np.max(abs(j))
        if j_m != 0:
            i = np.array(i)/j_m
            j = np.array(j)/j_m
        else:
            print("Max value 0 encountered")        
        signals_shift.append(i)
        Noisy_shift.append(j)
        mapping.append(j_m)
    
    print('Size of pure signal dataset', np.asarray(signals_shift).shape)
    print('Size of noisy signal dataset', np.asarray(Noisy_shift).shape)
    return np.asarray(signals_shift), np.asarray(Noisy_shift), mapping

# custom learning rate function
class LearningRateScheduler(Callback):
    def __init__(self, factor, patience):
        super(LearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.wait = 0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(K.get_value(self.model.optimizer.lr))
                new_lr = old_lr * self.factor
                K.set_value(self.model.optimizer.lr, new_lr)
                self.wait = 0
                print(f"Reducing learning rate to {new_lr}")


# print("Num GPUs Available: ", len(tf.test.gpu_device_name()))  ## attempted fix (also created error)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  ## created error
print("\n DeviceName = ", tf.test.gpu_device_name(), "\n Running Pgm:")

##############################################################################################  
parser = argparse.ArgumentParser()
parser.add_argument('--fil', nargs='+', type=int, help='Number of filters')
parser.add_argument('--ks', nargs='+', type=int, help='Size (length) of the kernel')
parser.add_argument('--layers', nargs='+', type=int, help='Number of sets of encoding and decoding layers')
args = parser.parse_args()

FIL = args.fil
KS = args.ks
Layers = args.layers

print("FIL = ", FIL)
print("KS = ", KS)
print("Layers = ", Layers)
##############################################################################################

ABS_PATH_HERE = f"/mnt/data/danakull/WaveformML/TrainingAndTesting/training/denoiser"       ## This is the path for titan
# ABS_PATH_HERE=f"/work/icecube/users/dkullgren/WaveformML/training/denoiser"               ## This is the path for caviness
dataset = 'Auger_v5'
# dataset = 'v6'

# Note that this is currently set up to use Run0 data only
DataDir = ABS_PATH_HERE + f"/../../data-production/taxi-noise/data/Dataset_{dataset}"   ## Enter directories here
# DataDir = ABS_PATH_HERE + f'/../../data-production/taxi-noise/data/Dataset_{dataset}'   ## Enter directories here
ModelsDir = ABS_PATH_HERE + f"/data/models/Denoiser_MSE_Updated_{dataset}"
LossAccDir = ABS_PATH_HERE + f"/data/loss_and_accuracy/Denoiser_MSE_Updated_{dataset}"
PlotsDir = ABS_PATH_HERE + f"/plots/Denoiser_MSE_Updated_{dataset}_Plots"

# Loading the Data:

# This script is able to load a small subset of the data to improve testing speed
# For good models, update the file names to the full datasets

# Uncomment for small files (directly from MakeTraces.py)
x_train = np.load(DataDir+'/pure_train.npy', allow_pickle=True)  ## Change files names
x_test = np.load(DataDir+'/pure_test.npy', allow_pickle=True)
x_train_noisy = np.load(DataDir+'/noisy_train.npy', allow_pickle=True)
x_test_noisy = np.load(DataDir+'/noisy_test.npy', allow_pickle=True)

# Uncomment for consolidated files
# x_train = np.load(DataDir+'/small_pure_train.npy')  ## Change files names
# x_test = np.load(DataDir+'/small_pure_test.npy')
# x_train_noisy = np.load(DataDir+'/small_noisy_train.npy')
# x_test_noisy = np.load(DataDir+'/small_noisy_test.npy')

# Remove noise only traces from noisy sets
train_sig_ind = [i for i in range(len(x_train)) if np.max(x_train[i]) != 0]
x_train = x_train[train_sig_ind]
x_train_noisy = x_train_noisy[train_sig_ind]

test_sig_ind = [i for i in range(len(x_test)) if np.max(x_test[i]) != 0]
x_test = x_test[test_sig_ind]
x_test_noisy = x_test_noisy[test_sig_ind]

# Preprocessing
x_train = np.array(x_train, dtype='float32')
x_test = np.array(x_test, dtype='float32')

x_train_noisy = np.array(x_train_noisy, dtype='float32')
x_test_noisy = np.array(x_test_noisy, dtype='float32')

# Normalizing in the range [-1,1]
x_train, x_train_noisy, MPT = Normalizing(x_train, x_train_noisy)
x_test, x_test_noisy, scalefactor_test = Normalizing(x_test, x_test_noisy)

# Shifting the peak locations
# x_train, x_train_noisy = ShiftTraces(x_train, x_train_noisy)
# x_test, x_test_noisy = ShiftTraces(x_test, x_test_noisy)

del MPT, scalefactor_test

##############################################################################################

# Model:
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def CreateModel(FIL, KS, Layers=2):
    # inputs = tf.keras.Input(shape=(4096,1))
    inputs = tf.keras.Input(shape=(1000,1))
    
    x = inputs
    for i in range(Layers):
        ## Encode
        x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
        x = MaxPooling1D(pool_size=2)(x)
    for i in range(Layers):
        ## Decode
        x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
        x = UpSampling1D(2)(x)
        
    # if Layers == 1:
    #     ## Encoder
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (inputs)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     ## Decode
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
        
    # elif Layers == 2:
    #     ## Encoder
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (inputs)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     ## Decode
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
        
    # elif Layers == 3:
    #     ## Encoder
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (inputs)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     ## Decode
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
        
    # elif Layers == 4:
    #     ## Encoder
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (inputs)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     ## Decode
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
        
    # elif Layers == 5:
    #     ## Encoder
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (inputs)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     ## Decode
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters= FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    
    # elif Layers == 6:
    #     ## Encoder
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (inputs)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     ## Decode
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    
    # elif Layers == 7:
    #     ## Encoder
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (inputs)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     ## Decode
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    
    # elif Layers == 8:
    #     ## Encoder
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (inputs)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = MaxPooling1D(pool_size=2)(x)
    #     ## Decode
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    #     x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    #     x = UpSampling1D(2)(x)
    
    output = Conv1D(filters=1, kernel_size=KS, padding='same', activation='linear') (x)
    Denoiser = Model(inputs, output)
    Denoiser.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=[CorrCoeff])
    # Denoiser.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=[CorrCoeff])
    
    return Denoiser

# FIL = [20]
# KS = [8, 16, 32, 64, 128, 256, 512]
for layer in Layers:
    for fil in FIL:
        for ks in KS:      
            print("\n Fil={0}, KS={1}, Layers={2} \n".format(fil, ks, layer))
            with strategy.scope():
                Denoiser = CreateModel(fil, ks, layer)
        
        #####################################################################
            lr_scheduler = LearningRateScheduler(factor=0.5, patience=5)       
            es = EarlyStopping(monitor='val_loss', mode='min',restore_best_weights=True, verbose=1, patience=5)
            mc = ModelCheckpoint(ModelsDir+f'/Denoiser_MSE_{dataset}_Fil={fil}_KS={ks}_Layers={layer}_v4.h5',
                        monitor = 'val_loss', mode = 'min', verbose=0, save_best_only=True)

        #####################################################################
        
            shape_0 = x_train.shape[0]
            shape_1 = x_train.shape[1]
            history = Denoiser.fit(np.reshape(x_train_noisy, (shape_0, shape_1, 1)), np.reshape(x_train, (shape_0, shape_1, 1)),
                                epochs=5000,
                                batch_size=200,
                                # batch_size=512,
                                shuffle=True,
                                validation_data=(x_test_noisy, x_test),
                                verbose=2,
                                callbacks=[lr_scheduler, es, mc]
                                )

            TrnLoss = np.array(history.history['loss'])
            TstLoss = np.array(history.history['val_loss'])
            # TrnAcc = np.array(history.history['CC2'])
            TrnAcc = np.array(history.history['CorrCoeff'])
            # TstAcc = np.array(history.history['val_CC2'])
            TstAcc = np.array(history.history['val_CorrCoeff'])
            np.save(LossAccDir + '/TrnLoss_Fil={0}_KS={1}_Layers={2}_v4.npy'.format(fil, ks, layer), TrnLoss)  #Change dir
            np.save(LossAccDir + '/TstLoss_Fil={0}_KS={1}_Layers={2}_v4.npy'.format(fil, ks, layer), TstLoss)


            NRows = 1
            NCols = 2
            gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
            fig = plt.figure(figsize=(6*NCols, 5*NRows))
            ax = fig.add_subplot(gs[0])

            ax.plot(TrnLoss, label='TrainLoss')
            ax.plot(TstLoss, label='TestLoss')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            ax.legend(loc='best', prop={'size': 8})

            ax = fig.add_subplot(gs[1])
            ax.plot(TrnAcc, label='Train_r')
            ax.plot(TstAcc, label='Test_r')
            ax.set_title("Fil={0}_KS={1}_Layers={2}".format(fil, ks, layer))
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Correlation Coefficient")
            ax.legend(loc='best', prop={'size': 8})
            fig.savefig(PlotsDir + '/DenoiserMSE_Fil={0}_KS={1}_Layers={2}_v4.pdf'.format(fil, ks, layer), bbox_inches='tight') #Change dir


print('***********************************')
print('***********************************')
print('***********************************')
print('************* FINISHING ***********')
print('***********************************')
print('***********************************')
print('***********************************')
print('***********************************')
print("ByeBye!")
