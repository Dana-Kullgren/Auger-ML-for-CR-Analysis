import numpy as np
import random
import os
from scipy.signal import hilbert
from icecube.icetray import I3Units

dataset = 'v21'
path = f"/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_{dataset}/"
channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]
cutoff = 0.005 # mV
CombineChannels = False

if CombineChannels:

	s = [[] for i in range(len(channels))]
	n = [[] for i in range(len(channels))]
	ns = [[] for i in range(len(channels))]

	# load traces
	for ich in range(len(channels)):
		ch = channels[ich]
		s[ich] = np.load(path + "Shifted_" + ch + "_Signals.npy")
		n[ich] = np.load(path + "Shifted_" + ch + "_NoiseOnly.npy")
		ns[ich] = np.load(path + "Shifted_" + ch + "_SigPlusNoise.npy")
		print(f'{ch} loaded')

	# calculate signal peaks
	s_new = [[] for j in range(len(channels))]
	n_new = [[] for j in range(len(channels))]
	ns_new = [[] for j in range(len(channels))]

	sigPeaks = []
	for ich in range(len(channels)):
		sigPeaks.append([0 for j in range(len(channels))])
		for i in range(len(s[ich])):
			sigPeaks[i][ich] = np.max(np.abs(hilbert(s[ich][i]))) / I3Units.mV
			if ich==0:
				sigPeaks.append([0 for j in range(len(channels))])

	sigPeaks = np.array(sigPeaks)
	print(f'np.shape(sigPeaks)={np.shape(sigPeaks)}')

	for ich in range(len(channels)):
		for i in range(len(s[ich])):
			greater_than_cutoff = np.all(sigPeaks[i] > cutoff)
			if greater_than_cutoff:
				s_new[ich].append(s[ich][i])
				n_new[ich].append(n[ich][i])
				ns_new[ich].append(ns[ich][i])

	for ich in range(len(channels)):
		ch = channels[ich]
		np.save(f"{path}CutOff={cutoff}_{ch}_Signals.npy", s_new[ich])
		np.save(f"{path}CutOff={cutoff}_{ch}_NoiseOnly.npy", n_new[ich])
		np.save(f"{path}CutOff={cutoff}_{ch}_SigPlusNoise.npy", ns_new[ich])

else:
	# load traces
	s = np.load(path + "Shifted_" + "Signals.npy")
	n = np.load(path + "Shifted_" + "NoiseOnly.npy")
	ns = np.load(path + "Shifted_" + "SigPlusNoise.npy")

	print(f'np.shape(s)={np.shape(s)}')

	# calculate signal peaks
	s_new = []
	n_new = []
	ns_new = []

	sigPeaks = [0 for i in range(len(s))]
	for i in range(len(s)):
		print(f'hilbert(s[i])={hilbert(s[i])}')
		sigPeaks[i] = np.max(np.abs(hilbert(s[i]))) / I3Units.mV

	sigPeaks = np.array(sigPeaks)
	print(f'np.shape(sigPeaks)={np.shape(sigPeaks)}')

	for i in range(len(s)):
		greater_than_cutoff = np.all(sigPeaks[i] > cutoff)
		if greater_than_cutoff:
			s_new.append(s[i])
			n_new.append(n[i])
			ns_new.append(ns[i])

	np.save(f"{path}CutOff={cutoff}_Signals.npy", s_new[ich])
	np.save(f"{path}CutOff={cutoff}_NoiseOnly.npy", n_new[ich])
	np.save(f"{path}CutOff={cutoff}_SigPlusNoise.npy", ns_new[ich])
