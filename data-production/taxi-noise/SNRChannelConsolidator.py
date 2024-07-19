import numpy as np
import os

ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
path =  ABS_PATH_HERE + "/data/Dataset_v17"

channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]

s = []
n = []
ns = []

for ich in range(len(channels)):
  channel = channels[ich]
  print(f'Loading {channel} files')

  new_s = np.load(path + f'/SNR_{channel}_Signals.npy')
  new_s = new_s.astype('float32')
  s.extend(new_s)

  new_n = np.load(path + f'/SNR_{channel}_NoiseOnly.npy')
  new_n = new_n.astype('float32')
  n.extend(new_n)

  new_ns = np.load(path + f'/SNR_{channel}_SigPlusNoise.npy')
  new_ns = new_ns.astype('float32')
  ns.extend(new_ns)

np.save(path + '/SNR_Signals.npy', s)
np.save(path + '/SNR_NoiseOnly.npy', n)
np.save(path + '/SNR_SigPlusNoise.npy', ns)