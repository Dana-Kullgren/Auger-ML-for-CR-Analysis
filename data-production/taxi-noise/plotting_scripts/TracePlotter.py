import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from icecube.icetray import I3Units
from scipy.signal import hilbert
import os

# Define function(s)
def GetSNR(Trace):
    '''
    Return the Signal to Noise Ratio. Signal is just the peak of the trace.
    Medain RMS of chunck of trace. 
    '''
    from scipy.signal import hilbert
    SigPeak = np.max(np.abs(hilbert(Trace))) # Can also use abs value instead of hilbert 
    Chunks = np.array_split(Trace, 10)  # Split the trace in to 10 small chunks
    ChunkRMS_squared = [(sum(chunk**2))/len(chunk) for chunk in Chunks] ## RMS^2 of each chunk
    RMS_Median = np.median(ChunkRMS_squared) ## Chunk with signal in it.
    return SigPeak**2/RMS_Median 

## The following function assumes that the signals are located at the center of traces
# def GetSNR(trace, binlow=510):
#   Trace_Peak = np.max(np.abs(trace)) 
#   chunk = np.array(trace[binlow:810])
#   RMS_squared = sum(chunk ** 2) / len(chunk)
#   SNR = Trace_Peak ** 2 / RMS_squared
#   return SNR


# Change this to the path for whatever file I need to access
dataset = 'Auger_v4'
path = f"/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_{dataset}/"

## List whichever type of waveforms you wish to plot as True and the rest as False
NoiseOnly = True
Signals = True
SigPlusNoise = True

ch_list = ["ant1ch0", "ant1ch1", "ant2ch0", "ant2ch1", "ant3ch0", "ant3ch1"]

s = [[] for i in range(len(ch_list))]
n = [[] for i in range(len(ch_list))]
ns = [[] for i in range(len(ch_list))]

for ich in range(len(ch_list)):
	ch = ch_list[ich]
	print("\n" + ch)

	if NoiseOnly:
		n[ich] = np.load(f"{path}Run_{ch}_NoiseOnly.npy", allow_pickle=True)
		print("Noise loaded")
		# noisefreq = np.load(path + "/Run1_Frequency_" + ch + "_NoiseOnly.npy")
		print("Number of traces in " + ch + ": Noise Only (Time)", len(n[ich]))
		# print("Noise Only (Frequency)", len(noisefreq))
	if Signals:
		s[ich] = np.load(f"{path}Run_{ch}_Signals.npy", allow_pickle=True)
		print("Signals loaded")
		print(f'np.shape(s) = {np.shape(s[ich])}')
		print("Number of traces in " + ch + ": Signal (Time)", len(s[ich]))
		# print("Signal (Frequency)", len(sigfreq))
	if SigPlusNoise:
		ns[ich] = np.load(f"{path}Run_{ch}_SigPlusNoise.npy", allow_pickle=True)
		print("Noisy signals loaded \n")
		# sigfreq = np.load(path + "/Run1_Frequency_" + ch + "_Signals.npy")
		# sigplusnoisefreq = np.load(path + "/Run1_Frequency_" + ch + "_SigPlusNoise.npy")
		print("Number of traces in " + ch + ": Signal Plus Noise (Time)", len(ns[ich]))
		# print("Signal Plus Noise (Frequency)", len(sigplusnoisefreq))

print(f'Signals: {s[0]}')
print(f'NoiseOnly: {n[0]}')
print(f'SigPlusNoise: {ns[0]}')


# Plotting starts here
NCols = 0
waveformTypes = 0	## this will be the number of waveform types (ie: Signals, NoiseOnly, SigPlusNoise) being plotted
plot_title = ''		## the title of the plot depends on the types of waveforms being plotted

if NoiseOnly:
	NCols += 1
	waveformTypes += 1
	plot_title += 'Noise_'
if Signals:
	NCols += 1
	# NCols += 2
	waveformTypes += 1
	plot_title += 'Signals_'
if SigPlusNoise:
	NCols += 1
	# NCols += 2
	waveformTypes += 1
	plot_title += 'SignalPlusNoise_'

NRows = len(ch_list)

plotindex = 0 # this is index of the trace I am plotting from the file (I chose 0 so I could plot once trace for each ant)
# while np.max(np.abs(hilbert(s[0][plotindex]))) / I3Units.mV > cutoff:
# 	plotindex += 1

# for plotindex in range(5):	## this is index of the trace I am plotting from the file
gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
fig = plt.figure(figsize=(6*NCols, 5*NRows))

print(f'NCols={NCols}')
print(f'NRows={NRows}')

for ich in range(len(ch_list)):
	ch = ch_list[ich]
	plot_position = 0 ## this will determine where the newest axis will be; increments after each axis
	sigPeak = np.max(np.abs(hilbert(s[ich][plotindex]))) / I3Units.mV # Can also use abs value instead of hilbert 
	# while sigPeak < 10.:
	# 	plotindex += 1
	# 	sigPeak = np.max(np.abs(hilbert(s[ich][plotindex]))) / I3Units.mV # Can also use abs value instead of hilbert 
	# 	print(f'plotindex={plotindex}, sigPeak={sigPeak}')

	if Signals:
		ax = fig.add_subplot(gs[plot_position+ich*waveformTypes])
		plot_position += 1
		print(f'ax[ich][0] = ax[{ich}][{0}]')
		x1 = np.array(s[ich][plotindex]) / I3Units.mV
		print(f'ch={ch}, plotindex={plotindex}, x1[0]={x1[0]}')
		ax.plot(x1, label = "Pure Signal")
		ax.set_xlabel("Time [ns]", fontsize=16)
		ax.set_ylabel(r"Amplitude [mV]", fontsize=16)
		ax.legend(loc='best', prop={'size': 12})
		# ax.set_title(f'{ch}, SigPeak={sigPeak:.3f}', fontsize=18) ## add units if end up using this
		ax.set_title(f'{ch}, SNR={GetSNR(s[ich][plotindex]):.3f}', fontsize=18)

	if NoiseOnly:
		ax = fig.add_subplot(gs[plot_position+ich*waveformTypes])
		plot_position += 1
		x2 = np.array(n[ich][plotindex]) / I3Units.mV
		ax.plot(x2, label = "Noise Only")
		ax.set_xlabel("Time [ns]", fontsize=16)
		ax.set_ylabel(r"Amplitude [mV]", fontsize=16)
		ax.legend(loc='best', prop={'size': 12})
		# ax.set_title(f'{ch}, SigPeak={sigPeak:.3f}', fontsize=18) ## add units if end up using this
		ax.set_title(f'{ch}, SNR={GetSNR(n[ich][plotindex]):.3f}', fontsize=18)

	if SigPlusNoise:
		ax = fig.add_subplot(gs[plot_position+ich*waveformTypes])
		plot_position += 1
		x3 = np.array(ns[ich][plotindex]) / I3Units.mV
		ax.plot(x3, label = "Signal Plus Noise")
		ax.set_xlabel("Time [ns]", fontsize=16)
		ax.set_ylabel(r"Amplitude [mV]", fontsize=16)
		ax.legend(loc='best', prop={'size': 12})
		# ax.set_title(f'{ch}, SigPeak={sigPeak:.3f}', fontsize=18) ## add units if end up using this
		ax.set_title(f'{ch}, SNR={GetSNR(ns[ich][plotindex]):.3f}', fontsize=18)

fig.suptitle('Traces', fontsize=24)  
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(path + f"/TracesPlot_{plot_title}{dataset}_plotindex={plotindex}.png", bbox_inches='tight')