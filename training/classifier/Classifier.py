import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
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
args = parser.parse_args()

KS = args.ks
Layers = args.layers
dl = args.dl
FIL = args.fil

##############################################################################################

# Loading the Data (already prepared):
# Change the dir
ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))

dataset = "Auger_v5"
# dataset = "v21.5"
DataDir = ABS_PATH_HERE + f"/../../data-production/taxi-noise/data/Dataset_{dataset}"   ## Enter directories here
ModelsDir = ABS_PATH_HERE + f"/data/models/Classifier_BCE_{dataset}"
LossAccDir = ABS_PATH_HERE + f"/data/loss_and_accuracy/Classifier_BCE_{dataset}"
PlotsDir = ABS_PATH_HERE + f"/plots/Classifier_BCE_{dataset}_Plots"

##############################################################################################

x_train = np.load(DataDir + "/traces_train.npy", allow_pickle=True)
x_test = np.load(DataDir + "/traces_test.npy", allow_pickle=True)
y_train = np.load(DataDir + "/labels_train.npy", allow_pickle=True)
y_test = np.load(DataDir + "/labels_test.npy", allow_pickle=True)

x_train = np.array(x_train, dtype='float32')
x_test = np.array(x_test, dtype='float32')
y_train = np.array(y_train, dtype='float32')
y_test = np.array(y_test, dtype='float32')

print(f'np.shape(x_train)={np.shape(x_train)}')

# x_train, MpTr = Normalization(x_train)
# x_test, MpTs = Normalization(x_test)

# x_train = ShiftTraces(x_train)
# x_test = ShiftTraces(x_test)

##############################################################################################

# if multiple GPUs are accessable, use this to run on multiple gpus. (2 gpus gives best results)
strategy = tf.distribute.MirroredStrategy()  
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Model
def CreateModel(KS, FIL, Layers, DLayers=1):
    inputs = tf.keras.Input(shape=(1000,1))

    x = inputs
    for i in range(Layers):
        ## Encoder
        x = Conv1D(filters=FIL, kernel_size=KS, padding='same', activation='relu') (x)
        x = MaxPooling1D(pool_size=2)(x)

    # x = Conv1D(filters=1, kernel_size=KS, padding='same', activation='relu', kernel_initializer='he_normal') (x)
    x = tf.keras.layers.Flatten()(x)
    if DLayers == 1:
        output = Dense(1, activation='sigmoid', kernel_initializer='he_normal') (x)
    if DLayers == 2:
        x = Dense(8, activation='relu', kernel_initializer='he_normal') (x)
        output = Dense(1, activation='sigmoid', kernel_initializer='he_normal') (x)
    if DLayers == 3:
        x = Dense(64, activation='relu', kernel_initializer='he_normal') (x)
        x = Dense(8, activation='relu',kernel_initializer='he_normal') (x)
        output = Dense(1, activation='sigmoid', kernel_initializer='he_normal') (x)
    Classifier = Model(inputs, output)
    print(Model.summary(Classifier))
    Classifier.compile(loss='bce', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    # Classifier.compile(loss='bce', optimizer=tf.keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
    return Classifier

# FIL = [20] # 8, 16, 32, 64
# KS = [8, 16, 32, 64] # 128, 256, 512
# Layers = [2] # 2, 3, 4, 5
# for fil in FIL:
# fil = "Default"
for fil in FIL: # Delete this line when I'm using different fil values in different layers
    for ks in KS:
        for layer in Layers:
            for DL in dl: # 1, 2, 3
                print("\n Fil={0}, KS={1}, DL={2} \n".format(fil, ks, DL))
                with strategy.scope():
                    Classifier = CreateModel(ks, fil, layer, DL)
                    # Classifier = CreateModel(ks, FIL_1, FIL_2, FIL_3, FIL_4, FIL_5, layer, DL)
                lr_scheduler = LearningRateScheduler(factor=0.5, patience=5)
                es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1, patience=5)
                # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                mc = ModelCheckpoint(ModelsDir+'/ClassifierBCE_Layers={0}_DL={1}_Fil={2}_KS={3}.h5'.format(layer, DL, fil, ks),
                                                    monitor = 'val_loss', mode = 'min', verbose=0, save_best_only=True)
                shape_0 = x_train.shape[0]
                shape_1 = x_train.shape[1]
                history = Classifier.fit(x = np.reshape(x_train, (shape_0, shape_1, 1)), y = y_train,
                                    epochs=5000,
                                    batch_size=300,
                                    # batch_size=512,
                                    shuffle=True,
                                    validation_data=(x_test, y_test),
                                    verbose=2,
                                    callbacks=[lr_scheduler, es, mc]
                                    )
                TrnAcc = np.array(history.history['accuracy'])
                TstAcc = np.array(history.history['val_accuracy'])
                TrnLoss = np.array(history.history['loss'])
                TstLoss = np.array(history.history['val_loss'])
                print("\n Layers = {0}, DL= {1}, Fil = {2} : KS = {3},".format(layer, DL, fil, ks))
                print("TrnAcc = {0}".format(max(TrnAcc)))
                print("TstAcc = {0}".format(max(TstAcc)))
                np.save(LossAccDir + '/TrnAcc_Layers={0}_DL={1}_Fil={2}_KS={3}.npy'.format(layer, DL, fil, ks), TrnAcc)
                np.save(LossAccDir + '/TstAcc_Layers={0}_DL={1}_Fil={2}_KS={3}.npy'.format(layer, DL, fil, ks), TstAcc)
                np.save(LossAccDir + '/TrnLoss_Layers={0}_DL={1}_Fil={2}_KS={3}.npy'.format(layer, DL, fil, ks), TrnLoss)
                np.save(LossAccDir + '/TstLoss_Layers={0}_DL={1}_Fil={2}_KS={3}.npy'.format(layer, DL, fil, ks), TstLoss)

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
                ax.set_title("Layers={0}_DL={1}_Fil={2}_KS={3}".format(layer, DL, fil, ks))
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Accuracy")
                ax.legend(loc='best', prop={'size': 8})
                fig.savefig(PlotsDir + f"/Classifier_{dataset}_BCE_Layers={layer}_DL={DL}_Fil={fil}_KS={ks}.pdf", bbox_inches='tight')

print("ByeBye!")
