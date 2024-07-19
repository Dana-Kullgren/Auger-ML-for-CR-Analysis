import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Run', type=str, help='Number of Run file')
args = parser.parse_args()
run_num = args.Run

ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
DataDir = ABS_PATH_HERE + "/taxi-noise/data/Dataset_v19"

# Uncomment for small files (directly from MakeTraces.py)
# NoisySignal = np.load(DataDir + "/Run" + run_num + "_SigPlusNoise.npy")
# PureSignal = np.load(DataDir + "/Run" + run_num + "_Signals.npy")
# snrNoisy = np.load(DataDir + "/SNR_Run" + run_num + "_Noisy_true.npy")

# Uncomment for consolidated files
NoisySignal = np.load(DataDir + '/SigPlusNoise.npy')
PureSignal = np.load(DataDir + '/Signals.npy')
snrNoisy = np.load(DataDir + '/SNRNoisy.npy')

##############################################################################################
from sklearn.model_selection import train_test_split
noisy_train, noisy_test, pure_train, pure_test, snrNoisy_train, snrNoisy_test = train_test_split(NoisySignal,
                                                             PureSignal, snrNoisy,
                                                             random_state=42,
                                                             test_size=0.2)


# Uncomment the following lines before running this script

np.save(DataDir + "/noisy_train.npy", noisy_train)
np.save(DataDir + "/noisy_test.npy", noisy_test)
np.save(DataDir + "/pure_train.npy", pure_train)
np.save(DataDir + "/pure_test.npy", pure_test)
np.save(DataDir + "/SNRNoisy_train.npy", snrNoisy_train)
np.save(DataDir + "/SNRNoisy_test.npy", snrNoisy_test)

# np.save(DataDir + "/Run" + run_num + "_noisy_train.npy", noisy_train)
# np.save(DataDir + "/Run" + run_num + "_noisy_test.npy", noisy_test)
# np.save(DataDir + "/Run" + run_num + "_pure_train.npy", pure_train)
# np.save(DataDir + "/Run" + run_num + "_pure_test.npy", pure_test)
# np.save(DataDir + "/SNR_Run" + run_num + "_Noisy_train_true.npy", snrNoisy_train)
# np.save(DataDir + "/SNR_Run" + run_num + "_Noisy_test_true.npy", snrNoisy_test)
