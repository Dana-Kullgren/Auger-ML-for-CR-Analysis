import numpy as np
import os

dataset = 'Auger_v2'
DataDir = f"/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_{dataset}/"
channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]
cutoff = 0.005 # mV

SigSum = 0
NoiseSum = 0
SigPlusNoiseSum = 0

for channel in channels:
	AllSignals = np.load(f"/{DataDir}{channel}_Signals.npy", allow_pickle=True)
	AllNoiseOnly = np.load(f"/{DataDir}{channel}_NoiseOnly.npy", allow_pickle=True)
	AllSigPlusNoise = np.load(f"/{DataDir}{channel}_SigPlusNoise.npy", allow_pickle=True)

	print("\nChannel: " + channel)
	print("Signals: ", np.shape(AllSignals))
	print("NoiseOnly: ", np.shape(AllNoiseOnly))
	print("SigPlusNoise: ", np.shape(AllSigPlusNoise))

	SigSum += np.shape(AllSignals)[0]
	NoiseSum += np.shape(AllNoiseOnly)[0]
	SigPlusNoiseSum += np.shape(AllSigPlusNoise)[0]

print("All Signals: ", SigSum)
print("All Noise Only: ", NoiseSum)
print("All SigPlusNoise: ", SigPlusNoiseSum)
