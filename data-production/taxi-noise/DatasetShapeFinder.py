import numpy as np
import matplotlib
import os

# Change this to the path for whatever file I need to access
ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
DataDir =  "/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_v23"

s = np.load(DataDir + "/TestRun_ant1ch0_Signals.npy", allow_pickle=True)
n = np.load(DataDir + "/TestRun_ant1ch0_NoiseOnly.npy", allow_pickle=True)
ns = np.load(DataDir + "/TestRun_ant1ch0_SigPlusNoise.npy", allow_pickle=True)

print(f'np.shape(s) = {np.shape(s)}')
print(f'np.shape(n) = {np.shape(n)}')
print(f'np.shape(ns) = {np.shape(ns)}')