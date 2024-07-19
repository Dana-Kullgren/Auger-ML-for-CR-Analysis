from I3Tray import I3Tray
from icecube import icetray, radcube, dataclasses
from icecube.radcube import defaults
from icecube.icetray.i3logging import log_info, log_error, log_fatal, log_trace
icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
import numpy as np
import os
from scipy.signal import hilbert

dataset = 'Auger_v3'
DataDir = f"/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_{dataset}"
ChannelSeparated = True

# Load signal
if ChannelSeparated:
  channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]
  snrTraces = [[] for ich in range(len(channels))]
  Signal = [[] for ich in range(len(channels))]
  Background =[[] for ich in range(len(channels))]
  SigNoise =[[] for ich in range(len(channels))]
  PureSig =[[] for ich in range(len(channels))]
  for ich in range(len(channels)):
    Signal[ich] = np.load(f'{DataDir}/{channels[ich]}_SigPlusNoise.npy', allow_pickle=True)
    Background[ich] = np.load(f'{DataDir}/{channels[ich]}_NoiseOnly.npy', allow_pickle=True)
    PureSig[ich] = np.load(f'{DataDir}/{channels[ich]}_Signals.npy', allow_pickle=True)
    ## Concatenating the signals and noise
    SigNoise[ich] = np.concatenate((Signal[ich], Background[ich]))

else:
  snrTraces = []
  Signal = np.load(DataDir + '/SigPlusNoise.npy')
  Background = np.load(DataDir + '/NoiseOnly.npy')
  PureSig = np.load(DataDir + '/Signals.npy')
  ## Concatenating the signals and noise
  SigNoise = np.concatenate((Signal, Background))


## Does this need to be updated to the SNR definition used in modules/SelectCleanSig.py ?
def SNR(Trace):
  SigPeak = np.max(np.abs(hilbert(Trace))) # Can also use abs value instead of hilbert 
  Chunks = np.array_split(Trace, 10)  # Split the trace in to 10 small chunks
  ChunkRMS_squared = [(sum(chunk**2))/len(chunk) for chunk in Chunks] ## RMS^2 of each chunk
  RMS_Median = np.median(ChunkRMS_squared) ## choose the median value from chunks
  return SigPeak**2/RMS_Median 

################################################################################################################

# Find the SNR value for each trace
if ChannelSeparated:
  for ich in range(len(channels)):
    print(f'\nchannel={channels[ich]}')
    for i in range(len(SigNoise[ich])):
      snrTraces[ich].append(SNR(SigNoise[ich][i]))
      print(f'{i}/{len(SigNoise[ich])}')
    snrTraces[ich] = np.array(snrTraces[ich])
    np.save(f"{DataDir}/SNRTraces_{channels[ich]}.npy", snrTraces[ich])

else:
  for i in range(len(SigNoise)):
    snrTraces.append(SNR(SigNoise[i]))
    print(f'{i}/{len(SigNoise)}')
  snrTraces = np.array(snrTraces)
  np.save(DataDir + "/SNRTraces.npy", snrTraces)