from I3Tray import I3Tray
from icecube import icetray, radcube, dataclasses
from icecube.radcube import defaults
from icecube.icetray.i3logging import log_info, log_error, log_fatal, log_trace
icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
import numpy as np
import os
from scipy.signal import hilbert

# Change this to the path for whatever files I need to access
ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
dataset = 'v21'
DataDir =  ABS_PATH_HERE + f"/data/Dataset_{dataset}"
 
def SNR(Trace):
  SigPeak = np.max(np.abs(hilbert(Trace))) # Can also use abs value instead of hilbert 
  Chunks = np.array_split(Trace, 10)  # Split the trace in to 10 small chunks
  ChunkRMS_squared = [(sum(chunk**2))/len(chunk) for chunk in Chunks] ## RMS^2 of each chunk
  RMS_Median = np.median(ChunkRMS_squared) ## choose the median value from chunks
  return SigPeak**2/RMS_Median 

snrTotalNoise = np.array([])
snrTotalSignals = np.array([])
snrTotalSigPlusNoise = np.array([])

# Load in channel data
channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]

for channel in channels:
  print("\nChannel: " + channel)
  s = np.load(DataDir + "/{0}_Signals.npy".format(channel))
  print("Signals loaded")
  n = np.load(DataDir + "/{0}_NoiseOnly.npy".format(channel))
  print("Noise loaded")
  ns = np.load(DataDir + "/{0}_SigPlusNoise.npy".format(channel))
  print("Noisy signals loaded \n")

  print("Number of traces in the files:")
  print("Signal", len(s))
  print("Noise Only", len(n))
  print("Signal Plus Noise", len(ns), "\n")

  snrSig = []
  snrNoise = []
  snrSigPlusNoise = []

  for i in range(len(ns)):
    snrSigPlusNoise.append(SNR(ns[i]))
    print(f'{i}/{len(ns)}')
  snrSigPlusNoise = np.array(snrSigPlusNoise)
  print("SigPlusNoise SNR calculated")

  for i in range(len(s)):
    snrSig.append(SNR(s[i]))
    print(f'{i}/{len(s)}')
  snrSig = np.array(snrSig)
  print("Signals SNR calculated")

  for i in range(len(n)):
    snrNoise.append(SNR(n[i]))
    print(f'{i}/{len(n)}')
  snrNoise = np.array(snrNoise)
  print("NoiseOnly SNR calculated")

  print(f'Mean SNR of background traces for {channel}: {np.mean(snrNoise)}')
  snrTotalNoise = np.append(snrTotalNoise, snrNoise)
  snrTotalSignals = np.append(snrTotalSignals, snrSig)
  snrTotalSigPlusNoise = np.append(snrSigPlusNoise, snrSigPlusNoise)

  np.save(DataDir + "/SNR_{0}_Signals.npy".format(channel), snrSig)
  np.save(DataDir + "/SNR_{0}_NoiseOnly.npy".format(channel), snrNoise)
  np.save(DataDir + "/SNR_{0}_SigPlusNoise.npy".format(channel), snrSigPlusNoise)


print(f'len(snrTotalSignals) = {len(snrTotalSignals)}')
print(f'len(snrTotalNoise) = {len(snrTotalNoise)}')
print(f'len(snrTotalSigPlusNoise) = {len(snrTotalSigPlusNoise)}')

np.save(DataDir + "/SNR_Signals.npy", snrTotalSignals)
np.save(DataDir + "/SNR_NoiseOnly.npy", snrTotalNoise)
np.save(DataDir + "/SNR_SigPlusNoise.npy", snrTotalSigPlusNoise)

print(f'Mean SNR of background traces for entire dataset: {np.mean(snrTotalNoise)}')
