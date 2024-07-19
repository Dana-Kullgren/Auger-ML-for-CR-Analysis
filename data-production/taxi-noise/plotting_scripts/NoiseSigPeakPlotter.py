import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from icecube.icetray import I3Units
from scipy.signal import hilbert
import os

# Change this to the path for whatever file I need to access
ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
dataset = 'v20'
path = ABS_PATH_HERE + "/../data/Dataset_" + dataset

channels = ["ant1ch0", "ant1ch1", "ant2ch0", "ant2ch1", "ant3ch0", "ant3ch1"]
sig_peaks = [[] for i in range(len(channels))]
background_peaks = [[] for i in range(len(channels))]

# Load in channel data
for ich in range(len(channels)):
	ch = channels[ich]
	Signals = np.load(path + "/Run_Time_" + ch + "_Signals.npy")
	Backgrounds = np.load(path + "/Run_Time_" + ch + "_NoiseOnly.npy")

	print(f'ch={ch}, len(Signals)={len(Signals)}')
	print(f'ch={ch}, len(Backgrounds)={len(Backgrounds)}')

	for itrace in range(len(Signals)):
		sig = Signals[itrace]
		sig = np.array(sig) / I3Units.mV # change units
		SigPeak = np.max(np.abs(hilbert(sig))) # Can also use abs value instead of hilbert 
		sig_peaks[ich].append(SigPeak)

	for itrace in range(len(Backgrounds)):
		background = Backgrounds[itrace]
		background = np.array(background) / I3Units.mV # change units
		BackgroundPeak = np.max(np.abs(hilbert(background)))
		background_peaks[ich].append(BackgroundPeak)

# print(f'sig_peaks={sig_peaks}')
# print(f'background_peaks={background_peaks}')

# Plotting
NRows = 2
NCols = 3

gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
fig = plt.figure(figsize=(5*NCols, 4*NRows))
fig.suptitle("Peaks of Traces")

for ich in range(len(channels)):
	ch = channels[ich]
	ax = fig.add_subplot(gs[ich])
	ax.hist(sig_peaks[ich], label="Signal")
	ax.hist(background_peaks[ich], label="Background")
	# ax.hist(sig_peaks[ich], bins = [.01*i for i in range(31)], label="Signal")
	# ax.hist(background_peaks[ich], bins = [.01*i for i in range(31)], label="Background")
	ax.set_xlabel(r"Peak Amplitude [mV]", fontsize='x-large')
	ax.legend(loc='best', prop={'size': 12})
	ax.set_title(ch, fontsize='x-large')

fig.savefig(path + f"/NoiseSigPeaks_{dataset}.pdf", bbox_inches='tight')