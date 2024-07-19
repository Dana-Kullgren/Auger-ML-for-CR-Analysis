import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import backend as K
import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os
##########################################################################################################################################

# this script uses 1D convolutions

def Normalization(signal):
    signals = np.asarray(signal)
    # Normalizing traces in the range [-1, 1]
    Normalized_sig = []
    ScaleFactor = []
    for sig in signals:
        i_m = np.max(abs(sig))
        if i_m != 0:
            NormSIg = np.array(sig) / i_m
            Normalized_sig.append(NormSIg)
        else:
            print("Max value 0 encountered")
        ScaleFactor.append(i_m)
    print('Size of dataset', np.asarray(Normalized_sig).shape)
    return np.asarray(Normalized_sig), ScaleFactor

# def ShiftTraces(Sig):
#     # Takes all Sig traces and roll them
#     import random
#     shiftedSig = []
#     for i in range(len(Sig)):
#         shifInd = random.randint(-1500,1500) # shifting by random no of bins
#         shif =  np.roll(Sig[i], shifInd)
#         shiftedSig.append(shif)
#     shiftedSig = np.array(shiftedSig)
#     return shiftedSig

def ShiftTraces(Sig):
    # Takes all Sig traces and roll them (to move the peak locations)
    import random
    shiftedSig = []
    print(f'len(Sig[0]) = {len(Sig[0])}, should be 1000')
    for i in range(len(Sig)):
        spread = int(len(Sig[0])*.25)
        shift = random.randint(-spread,spread) # shifting by random number of bins
        shif =  np.roll(Sig[i], shift)
        shiftedSig.append(shif)
    shiftedSig = np.array(shiftedSig)
    return shiftedSig

# def Normalization(signal): # if want to Normalize in the range [0, 1]
#     signals = np.asarray(signal)
#     # Normalization
#     Normalized_sig = []
#     ScaleFactor = []
#     for sig in signals:
#         i_m = np.max(abs(sig))
#         NormSIg = (np.array(sig) / (2*i_m) ) + 0.5
#         Normalized_sig.append(NormSIg)
#         ScaleFactor.append(i_m)
#     print('Size of dataset', np.asarray(Normalized_sig).shape)
#     return np.asarray(Normalized_sig), ScaleFactor

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

##########################################################################################################################################
## Include this for the ouput file to look cleaner
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

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

## See number of GPU available to run the job
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("\n DeviceName = ", tf.test.gpu_device_name(), "\n Running Pgm: \n\n")

##############################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--ks', type=int, nargs='+', help='Size (length) of the kernel')
parser.add_argument('--layers', type=int, nargs='+', help='Number of sets of encoding and decoding layers')
parser.add_argument('--dl', type=int, nargs='+', help='Number of dense layers')
parser.add_argument('--fil', type=int, nargs='+', help='Number of filters in the outer layer of the network')
parser.add_argument('--ant', nargs='+', type=str, help='Name of antenna (ex: ant1)')
args = parser.parse_args()

FIL = args.fil
KS = args.ks
Layers = args.layers
ANT = args.ant[0]
dl = args.dl

print("FIL = ", FIL)
print("KS = ", KS)
print("Layers = ", Layers)
print("ANT = ", ANT)
print("dl = ", dl)
##############################################################################################

ABS_PATH_HERE=f"/home/danakull/work/WaveformML/TrainingAndTesting/training/classifier"
# dataset = 'v21.5'
dataset = 'Auger_v5'
# lr = 5e-3
NeedFileReformat = False
lr = 1e-4

DataDir = ABS_PATH_HERE + f'/../../data-production/taxi-noise/data/Dataset_{dataset}'   ## Enter directories here
ModelsDir = ABS_PATH_HERE + f"/data/models/Classifier_MSE_{dataset}"
LossAccDir = ABS_PATH_HERE + f"/data/loss_and_accuracy/Classifier_MSE_{dataset}"
PlotsDir = ABS_PATH_HERE + f"/plots/Classifier_MSE_{dataset}_Plots"

##############################################################################################

x_train = np.load(f"{DataDir}/{ANT}_traces_train.npy", allow_pickle=True)
x_test = np.load(f"{DataDir}/{ANT}_traces_test.npy", allow_pickle=True)
y_train = np.load(f"{DataDir}/{ANT}_labels_train.npy", allow_pickle=True)
y_test = np.load(f"{DataDir}/{ANT}_labels_test.npy", allow_pickle=True)

x_train = np.array(x_train, dtype='float32')
x_test = np.array(x_test, dtype='float32')
y_train = np.array(y_train, dtype='float32')
y_test = np.array(y_test, dtype='float32')

# y_train = y_train.reshape((y_train.shape[0], 1))
# y_test = y_test.reshape((y_test.shape[0], 1))

x_train, MpTr = Normalization(x_train)
x_test, MpTs = Normalization(x_test)

normalizers_train = np.array(MpTr)
normalizers_test = np.array(MpTs)

np.save(f'{DataDir}/{ANT}_classifier_normalizers_train.npy', normalizers_train)
np.save(f'{DataDir}/{ANT}_classifier_normalizers_test.npy', normalizers_test)

print('Normalizers saved')

print(f'x_train.shape[0] = {x_train.shape[0]}')
print(f'x_train.shape[1] = {x_train.shape[1]}')
print(f'x_train.shape[2] = {x_train.shape[2]}\n')

print(f'x_test.shape[0] = {x_test.shape[0]}')
print(f'x_test.shape[1] = {x_test.shape[1]}')
print(f'x_test.shape[2] = {x_test.shape[2]}\n')

print(f'y_train.shape[0] = {y_train.shape[0]}')
print(f'y_train.shape[1] = {y_train.shape[1]}\n')

print(f'y_test.shape[0] = {y_test.shape[0]}')
print(f'y_test.shape[1] = {y_test.shape[1]}\n')

# x_train = ShiftTraces(x_train)
# x_test = ShiftTraces(x_test)

if NeedFileReformat:
    x_train = np.reshape(x_train, (x_train.shape[1], x_train.shape[2], x_train.shape[0]))
    x_test = np.reshape(x_test, (x_test.shape[1], x_test.shape[2], x_test.shape[0]))
    y_train = np.reshape(y_train, (y_train.shape[1], y_train.shape[0]))
    y_test = np.reshape(y_test, (y_test.shape[1], y_test.shape[0]))


##############################################################################################

# if multiple GPUs are accessable, use this to run on multiple gpus. (2 gpus gives best results)
strategy = tf.distribute.MirroredStrategy()  
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Model
def CreateModel(KS, FIL, Layers, DLayers=1):

    # Define the input shape for each polarization
    input_shape = (1000, 2)  # 2 channels, 1 for each polarization

    # Define the input layer
    input_layer = Input(shape=input_shape)
    print(f'np.shape(input_layer) = {np.shape(input_layer)}')
    x = input_layer

    # Convolutional layers
    for i in range(Layers):
        x = Conv1D(filters=FIL, kernel_size=KS, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
    # x = Conv1D(filters=FIL, kernel_size=KS, activation="relu", padding='same')(x)

    # Flatten the pooled output
    x = Flatten()(x)
    
    # activation = 'relu'
    for i in range(DLayers):
        if (DLayers-i) == 1:
            activation='sigmoid'
            output_dim = 1
        elif (DLayers-i) == 2:
            activation = 'relu'
            output_dim = 2
        elif (DLayers-i) == 3:
            activation = 'relu'
            output_dim = 4
        # else:
        # # if ((DLayers-i) == 1) or ((DLayers-i) == 2):
        #     output_dim = 2
        #     activation = 'relu'
        x = Dense(output_dim, activation=activation)(x)
    output = x

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['accuracy'])
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='bce', metrics=['accuracy'])
    return model

# FIL = [20] # 8, 16, 32, 64
# KS = [8, 16, 32, 64] # 128, 256, 512
# Layers = [2] # 2, 3, 4, 5
# for fil in FIL:
# fil = "Default"

# --layers 4 --dl 1

for fil in FIL: # Delete this line when I'm using different fil values in different layers
    for ks in KS:
        for layer in Layers:
            for DL in dl: # 1, 2, 3
                print("\n Fil={0}, KS={1}, DL={2} \n".format(fil, ks, DL))
                with strategy.scope():
                    Classifier = CreateModel(ks, fil, layer, DL)
                    # Classifier = CreateModel(ks, FIL_1, FIL_2, FIL_3, FIL_4, FIL_5, layer, DL)
                # lr_scheduler = LearningRateScheduler(factor=0.5, patience=5)
                # es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1, patience=5)
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                mc = ModelCheckpoint(ModelsDir+f'/ClassifierMSE_Layers={layer}_DL={DL}_Fil={fil}_KS={ks}_ANT={ANT}_lr={lr}.h5',
                                                    monitor = 'val_loss', mode = 'min', verbose=0, save_best_only=True)
                shape_0 = x_train.shape[0]
                shape_1 = x_train.shape[1]
                shape_2 = x_train.shape[2]
                # print(np.shape(x_train))
                print(np.shape(y_train))
                print(np.shape(x_test))
                print(np.shape(y_test))
                history = Classifier.fit(np.reshape(x_train, (shape_0, shape_1, shape_2)), np.reshape(y_train, (shape_0, shape_2)),
                # history = Classifier.fit(x_train, y_train,
                                    epochs=5000,
                                    
                                    # 7/8/24 updated batch size
                                    # batch_size=300,
                                    batch_size=512,

                                    shuffle=True,
                                    validation_data=(x_test, y_test),
                                    verbose=2,
                                    callbacks=[es, mc]
                                    # callbacks=[lr_scheduler,es, mc]
                                    )
                TrnAcc = np.array(history.history['accuracy'])
                TstAcc = np.array(history.history['val_accuracy'])
                TrnLoss = np.array(history.history['loss'])
                TstLoss = np.array(history.history['val_loss'])
                print(f"\n Layers = {layer}, DL= {DL}, Fil = {fil}, KS = {ks}, ANT = {ANT}")
                print("TrnAcc = {0}".format(max(TrnAcc)))
                print("TstAcc = {0}".format(max(TstAcc)))
                np.save(LossAccDir + f'/TrnAcc_Layers={layer}_DL={DL}_Fil={fil}_KS={ks}_ANT={ANT}_lr={lr}.npy', TrnAcc)
                np.save(LossAccDir + f'/TstAcc_Layers={layer}_DL={DL}_Fil={fil}_KS={ks}_ANT={ANT}_lr={lr}.npy', TstAcc)
                np.save(LossAccDir + f'/TrnLoss_Layers={layer}_DL={DL}_Fil={fil}_KS={ks}_ANT={ANT}_lr={lr}.npy', TrnLoss)
                np.save(LossAccDir + f'/TstLoss_Layers={layer}_DL={DL}_Fil={fil}_KS={ks}_ANT={ANT}_lr={lr}.npy', TstLoss)

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
                ax.plot(TrnAcc, label='TrainAccuracy')
                ax.plot(TstAcc, label='TestAccuracy')
                ax.set_title(f"Layers={layer}_DL={DL}_Fil={fil}_KS={ks}_ANT={ANT}_lr={lr}")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Accuracy")
                ax.legend(loc='best', prop={'size': 8})
                fig.savefig(PlotsDir + f"/Classifier_{dataset}_MSE_Layers={layer}_DL={DL}_Fil={fil}_KS={ks}_ANT={ANT}_lr={lr}.pdf", bbox_inches='tight')

print("ByeBye!")

# #If you want to use 1D CNN on both channels individually then combine them at the end, try something like this:

# from tensorflow.keras import layers, Model

# # Define the input shape for each polarization
# input_shape = (1000,)

# # Define the input layers for each polarization
# input_polarization_1 = layers.Input(shape=input_shape)
# input_polarization_2 = layers.Input(shape=input_shape)

# # Reshape the inputs to include the channel dimension
# reshaped_polarization_1 = layers.Reshape(input_shape + (1,))(input_polarization_1)
# reshaped_polarization_2 = layers.Reshape(input_shape + (1,))(input_polarization_2)

# # Convolutional layers for each polarization
# conv_polarization_1 = layers.Conv1D(32, kernel_size=3, activation='relu')(reshaped_polarization_1)
# conv_polarization_1 = layers.MaxPooling1D(pool_size=2)(conv_polarization_1)
# conv_polarization_1 = layers.Flatten()(conv_polarization_1)

# conv_polarization_2 = layers.Conv1D(32, kernel_size=3, activation='relu')(reshaped_polarization_2)
# conv_polarization_2 = layers.MaxPooling1D(pool_size=2)(conv_polarization_2)
# conv_polarization_2 = layers.Flatten()(conv_polarization_2)

# # Concatenate the outputs of the convolutional layers
# merged = layers.concatenate([conv_polarization_1, conv_polarization_2])

# # Dense layers
# dense = layers.Dense(64, activation='relu')(merged)
# output = layers.Dense(1, activation='sigmoid')(dense)

# # Create the model
# model = Model(inputs=[input_polarization_1, input_polarization_2], outputs=output)

# # Compile the model
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])