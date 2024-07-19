from I3Tray import I3Tray
from icecube import icetray, radcube, dataclasses
from icecube.radcube import defaults
from icecube.icetray.i3logging import log_info, log_error, log_fatal, log_trace
icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
import numpy as np
import os
import argparse
from scipy.signal import hilbert

parser = argparse.ArgumentParser()
parser.add_argument('--Run', type=str, help='Number of Run file')
args = parser.parse_args()
run_num = args.Run

# Change this to the path for whatever files I need to access
dataset = 'Auger_v3'
path = f"/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_{dataset}"
ChannelSeparated = True

# Uncomment for small files (directly from MakeTraces.py)
# Noisy = np.load(path + "/Run" + run_num + "_SigPlusNoise.npy")
# print("Noisy data loaded")
# PureSig = np.load(path + "/Run" + run_num + "_Signals.npy")
# print("Pure signal data loaded")

# Uncomment for consolidated files
if ChannelSeparated:
  channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]
  snrNoisy = [[] for ich in range(len(channels))]
  Noisy = [[] for ich in range(len(channels))]
  PureSig =[[] for ich in range(len(channels))]
  for ich in range(len(channels)):
    Noisy[ich] = np.load(f"{path}/{channels[ich]}_SigPlusNoise.npy",allow_pickle=True)
    print("Noisy data loaded")
    PureSig[ich] = np.load(f"{path}/{channels[ich]}_Signals.npy",allow_pickle=True)
    print("Pure signal data loaded")
else:
  snrNoisy = []
  Noisy = np.load(f"{path}/SigPlusNoise.npy")
  print("Noisy data loaded")
  PureSig = np.load(f"{path}/Signals.npy")
  print("Pure signal data loaded")

## Does this need to be updated to the SNR definition used in modules/SelectCleanSig.py ?
def SNR(Trace):
  SigPeak = np.max(np.abs(hilbert(Trace))) # Can also use abs value instead of hilbert 
  Chunks = np.array_split(Trace, 10)  # Split the trace in to 10 small chunks
  ChunkRMS_squared = [(sum(chunk**2))/len(chunk) for chunk in Chunks] ## RMS^2 of each chunk
  RMS_Median = np.median(ChunkRMS_squared) ## choose the median value from chunks
  return SigPeak**2/RMS_Median 

###########################################################################################################

# Measured SNR (no longer used)
# for i in range(len(Noisy)):
#     if i % 500 == 0:
#       print(i, "/", len(Noisy), "   ", int(i/len(Noisy)*100))
#     signalMiddle = SignalMiddleFinder(PureSig[i])
#     signalStart, signalStop, noiseStart, noiseStop = GetSignalNoiseWindows(signalMiddle)
#     snrNoisy.append(SNR(Noisy[i], signalStart, signalStop, noiseStart, noiseStop))
    
# True SNR (no longer used)
# for i in range(len(Noisy)):
#     if i % 500 == 0:
#       print(i, "/", len(Noisy), "   ", int(i/len(Noisy)*100))
#     snrNoisy.append(SNR(Noisy[i]))
#     # signalMiddle = SignalMiddleFinder(PureSig[i])
#     # signalStart, signalStop, noiseStart, noiseStop = GetSignalNoiseWindows(signalMiddle)
#     # snrNoisy.append(SNR(Noisy[i], PureSig[i], signalStart, signalStop, noiseStart, noiseStop))
# snrNoisy = np.array(snrNoisy)

if ChannelSeparated:
  for ich in range(len(channels)):
    print(f'\nchannel={channels[ich]}')
    for i in range(len(Noisy[ich])):
        snrNoisy[ich].append(SNR(Noisy[ich][i]))
        print(f'{i}/{len(Noisy[ich])}')
    snrNoisy[ich] = np.array(snrNoisy[ich])
    print(f'len(snrNoisy[ich])={len(snrNoisy[ich])}')
    np.save(f"{path}/SNRNoisy_{channels[ich]}.npy", snrNoisy[ich])

else:
  for i in range(len(Noisy)):
      snrNoisy.append(SNR(Noisy[i]))
      print(f'{i}/{len(Noisy)}')
  snrNoisy = np.array(snrNoisy)
  np.save(path + "/SNRNoisy.npy", snrNoisy)