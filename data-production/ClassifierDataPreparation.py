import numpy as np
import os

ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
DataDir = ABS_PATH_HERE + "/taxi-noise/data/Dataset_v19"

Signal = np.load(DataDir + f'/SigPlusNoise.npy')
print(f'Signal: {Signal.shape}')
Background = np.load(DataDir + f'/NoiseOnly.npy')
print(f'Background: {Background.shape}')
snrTraces = np.load(DataDir + f"/SNRTraces.npy")
print(f'snrTraces: {snrTraces.shape}')
print('Files loaded')

## Concatenating the signals and noise
SigNoise = np.concatenate((Signal, Background))

# Creating Labels
LabelS = np.ones(len(Signal)) # This is SigPlusNoise
LabelN = np.zeros(len(Background))
Label = np.concatenate((LabelS, LabelN))
print("Labels created")
##############################################################################################
from sklearn.model_selection import train_test_split
traces_train, traces_test, labels_train, labels_test, snrTraces_train, snrTraces_test = train_test_split(SigNoise,
                                                             Label, snrTraces,
                                                             random_state=42,
                                                             test_size=0.2)

np.save(DataDir + "/traces_train.npy", traces_train)
np.save(DataDir + "/traces_test.npy", traces_test)
np.save(DataDir + "/labels_train.npy", labels_train)
np.save(DataDir + "/labels_test.npy", labels_test)
np.save(DataDir + "/SNRTraces_train.npy", snrTraces_train)
np.save(DataDir + "/SNRTraces_test.npy", snrTraces_test)
print('Files saved')