import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
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
#         correlation = cov_xy / (std_x * std_y)
#         return correlation

#     correlation = tf.py_function(pearson_correlation, [y_true, y_pred], tf.float32)
#     return correlation


def ShiftTraces(Sig, Noisy):
    # Takes all Sig and Noisy traces and roll them (to move the peak locations)
    import random
    shiftedSig = []
    shiftedNoisy = []
    print(f'len(Sig[0]) = {len(Sig[0])}, should be 1000')
    for i in range(len(Sig)):
        spread = int(len(Sig[0])*.25)
        shift = random.randint(-spread,spread) # shifting by random number of bins

        shif =  np.roll(Sig[i], shift)
        shifN = np.roll(Noisy[i], shift)

        shiftedSig.append(shif)
        shiftedNoisy.append(shifN)

    shiftedSig = np.array(shiftedSig)
    shiftedNoisy = np.array(shiftedNoisy)
    return shiftedSig, shiftedNoisy

def Normalizing(signal, noised_signal): ## Normalize by the peak of Noisy traces.
    Signals = np.asarray(signal)
    NoisySig = np.asarray(noised_signal)
    # Normalization
    signals_shift = []
    Noisy_shift = []
    mapping = []
    for itrc in range(len(Signals)):
        noisy_max = np.max(abs(NoisySig[itrc]))
        if noisy_max != 0:
            sig = np.array(Signals[itrc]) / noisy_max
            noisy = np.array(NoisySig[itrc]) / noisy_max
        else:
            print("Max value 0 encountered")
        signals_shift.append(sig)
        Noisy_shift.append(noisy)
        mapping.append(noisy_max)
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
parser.add_argument('--ant', nargs='+', type=str, help='Name of antenna (ex: ant1)')
args = parser.parse_args()

FIL = args.fil
KS = args.ks
Layers = args.layers
ANT = args.ant[0]

print("FIL = ", FIL)
print("KS = ", KS)
print("Layers = ", Layers)
print("ANT = ", ANT)
##############################################################################################

ABS_PATH_HERE=f"/home/danakull/work/WaveformML/TrainingAndTesting/training/denoiser"
dataset = 'Auger_v5'
# dataset = 'v21.5'
lr = 1e-3
NeedFileReformat = False  ## this will reformat files with the shape (2, # traces, 1000) to (# traces, 1000, 2)

# Note that this is currently set up to use Run0 data only
DataDir = ABS_PATH_HERE + f'/../../data-production/taxi-noise/data/Dataset_{dataset}'   ## Enter directories here
ModelsDir = ABS_PATH_HERE + f"/data/models/Denoiser_MSE_Updated_{dataset}"
LossAccDir = ABS_PATH_HERE + f"/data/loss_and_accuracy/Denoiser_MSE_Updated_{dataset}"
PlotsDir = ABS_PATH_HERE + f"/plots/Denoiser_MSE_Updated_{dataset}_Plots"

# Loading the Data:
x_train = np.load(f'{DataDir}/{ANT}_pure_train.npy', allow_pickle=True)  ## Change files names
x_test = np.load(f'{DataDir}/{ANT}_pure_test.npy', allow_pickle=True)
x_train_noisy = np.load(f'{DataDir}/{ANT}_noisy_train.npy', allow_pickle=True)
x_test_noisy = np.load(f'{DataDir}/{ANT}_noisy_test.npy', allow_pickle=True)

# print(f'x_train={x_train}')

if NeedFileReformat:
    x_train = np.reshape(x_train, (x_train.shape[1], x_train.shape[2], x_train.shape[0]))
    x_test = np.reshape(x_test, (x_test.shape[1], x_test.shape[2], x_test.shape[0]))
    x_train_noisy = np.reshape(x_train_noisy, (x_train_noisy.shape[1], x_train_noisy.shape[2], x_train_noisy.shape[0]))
    x_test_noisy = np.reshape(x_test_noisy, (x_test_noisy.shape[1], x_test_noisy.shape[2], x_test_noisy.shape[0]))

print(f'x_train={np.shape(x_train)}')
print(f'x_test={np.shape(x_test)}')
print(f'x_train_noisy={np.shape(x_train_noisy)}')
print(f'x_test_noisy={np.shape(x_test_noisy)}')

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

normalizers_train = np.array(MPT)
normalizers_test = np.array(scalefactor_test)

np.save(f'{DataDir}/{ANT}_denoiser_normalizers_train.npy', normalizers_train)
np.save(f'{DataDir}/{ANT}_denoiser_normalizers_test.npy', normalizers_test)

print('Normalizers saved')

del MPT, scalefactor_test

print(f'np.shape(x_train)={np.shape(x_train)}')
print(f'np.shape(x_train_noisy)={np.shape(x_train_noisy)}')

##############################################################################################

# Model:
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def CreateModel(FIL, KS, Layers=2):
    # Define the input shape for each polarization
    input_shape = (1000, 2)
    
    input_layer = Input(shape=input_shape)
    x = input_layer
    print(f'Shape of input: {np.shape(x)}')

    # encoding section
    for i in range(Layers):
        x = Conv1D(filters=FIL, kernel_size=KS, activation='relu', padding='same')(x)
        print(f'Shape after Conv1D: {np.shape(x)}')
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        print(f'Shape after MaxPooling1D: {np.shape(x)}')
    # x = Conv1D(filters=FIL, kernel_size=KS, activation='relu', padding='same')(x)

   # decoding section
    for i in range(Layers):
        x = Conv1D(filters=FIL, kernel_size=KS, activation='relu', padding='same')(x)
        print(f'Shape after Conv1D: {np.shape(x)}')
        x = UpSampling1D(2)(x)
        print(f'Shape after UpSampling1D: {np.shape(x)}')
    output = Conv1D(2, kernel_size=KS, activation='linear', padding='same')(x)
    # output = Conv1D(1, kernel_size=KS, activation='linear', padding='same')(x)
    print(f'Shape of output: {np.shape(output)}')
    # output = Conv1D(filters=FIL, kernel_size=KS, activation='linear', padding='same')(x)

   #  # encoding section
   #  x = Conv1D(32, 3, activation='relu', padding='same')(x)
   #  x = MaxPooling1D(2, padding='same')(x)
   #  x = Conv1D(64, 3, activation='relu', padding='same')(x)

   # # decoding section
   #  x = Conv1D(64, 3, activation='relu', padding='same')(x)
   #  x = UpSampling1D(2)(x)
   #  output = Conv1D(2, 3, activation='linear', padding='same')(x)

    # Create the autoencoder model
    autoencoder = Model(input_layer, output)

    # Compile the model
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=[CorrCoeff])
    # autoencoder.fit(x_train, y_train, epochs=10, batch_size=500, validation_data=(x_test, y_test))

    return autoencoder

for layer in Layers:
    for fil in FIL:
        for ks in KS:      
            print("\n Fil={0}, KS={1}, Layers={2} \n".format(fil, ks, layer))
            with strategy.scope():
                Denoiser = CreateModel(fil, ks, layer)
        
        #####################################################################
            lr_scheduler = LearningRateScheduler(factor=0.5, patience=5)
            es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1, patience=5)
            mc = ModelCheckpoint(f'{ModelsDir}/Denoiser_MSE_{dataset}_Fil={fil}_KS={ks}_Layers={layer}_ANT={ANT}.h5',
                        monitor = 'val_loss', mode = 'min', verbose=0, save_best_only=True)

        #####################################################################
        
            print('\nAfter training is complete:')
            print(f'np.shape(x_train)={np.shape(x_train)}')
            print(f'np.shape(x_train_noisy)={np.shape(x_train_noisy)}')

            shape_0 = x_train.shape[0]
            shape_1 = x_train.shape[1]
            shape_2 = x_train.shape[2]

            print(f'shape_0 = {shape_0}')
            print(f'shape_1 = {shape_1}')
            print(f'shape_2 = {shape_2}')

            history = Denoiser.fit(np.reshape(x_train_noisy, (shape_0, shape_1, shape_2)), np.reshape(x_train, (shape_0, shape_1, shape_2)),
                                epochs=5000,
                                batch_size=500,
                                shuffle=True,
                                validation_data=(x_test_noisy, x_test),
                                verbose=2,
                                callbacks=[lr_scheduler, es, mc]
                                )

            DenoisedWaveforms = Denoiser.predict(x_train_noisy)
            print(f'np.shape(DenoisedWaveforms)={np.shape(DenoisedWaveforms)}')

            print(f'history.history.keys() = {history.history.keys()}')
            TrnLoss = np.array(history.history['loss'])
            TstLoss = np.array(history.history['val_loss'])
            TrnAcc = np.array(history.history['CorrCoeff'])
            TstAcc = np.array(history.history['val_CorrCoeff'])
            np.save(LossAccDir + f'/TrnLoss_Fil={fil}_KS={ks}_Layers={layer}_ANT={ANT}.npy', TrnLoss)  #Change dir
            np.save(LossAccDir + f'/TstLoss_Fil={fil}_KS={ks}_Layers={layer}_ANT={ANT}.npy', TstLoss)

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
            fig.savefig(PlotsDir + f'/DenoiserMSE_Fil={fil}_KS={ks}_Layers={layer}_ANT={ANT}.pdf', bbox_inches='tight') #Change dir


print('***********************************')
print('***********************************')
print('***********************************')
print('************* FINISHING ***********')
print('***********************************')
print('***********************************')
print('***********************************')
print('***********************************')
print("ByeBye!")
