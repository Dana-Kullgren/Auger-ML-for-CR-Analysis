import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statistics as stat
import os
from I3Tray import I3Tray
from icecube import icetray, radcube, dataclasses, taxi_reader
from icecube.radcube import defaults
from icecube.icetray import I3Units

# change this to the path for whatever file I need to access
ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
dataset = 'v21'
path = ABS_PATH_HERE + f"/../data/Dataset_{dataset}"

RMSFileGiven = False
SeparateChannels = True
SeparateByCascading = True
elec_resp = ""

channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]

def GetRMS(NoiseOnly, RMSList):
	for trace in NoiseOnly:
	  peak, rms, snr = radcube.GetChunkSNR(trace, No_Chunks=10)
	  rms = rms/I3Units.mV
	  RMSList = np.append(RMSList, rms)
	return RMSList

# Load in data
if RMSFileGiven:
	if SeparateChannels:
		channels_RMS = np.load(path + f'/{elec_resp}Run_RMS.npy', allow_pickle=True)
		# channels_RMS = [[] for i in range(len(channels))]
		# for i in range(len(channels)):
			# channels_RMS[i] = np.load(path + f'/{channels[i]}_RMS.npy', allow_pickle=True)
		# rms = np.load(path + f"/RMS.npy", allow_pickle=True)
	else:
		rms = []
		for j in range(10):
			rms_path = np.load(path + f"/{elec_resp}Run{j}_RMS.npy", allow_pickle=True)
			rms.extend(rms_path)
else:
	if SeparateChannels:
		noise = [[] for i in range(len(channels))]
		for i in range(len(channels)):
			noise[i] = np.load(path + f"/{channels[i]}_NoiseOnly.npy", allow_pickle=True)

		# Find RMS values for NoiseOnlyTraces
		RMSList = [[] for i in range(len(channels))]
		for i in range(len(channels)):
			RMSList[i] = GetRMS(noise[i], RMSList[i]) # rms is returned in mV
			print(f'{channels[i]} SNR calculated')
	else:
		noise = np.load(path + f'/NoiseOnly.npy', allow_pickle=True)
		RMSList = []
		RMSList = GetRMS(noise, RMSList) # rms is returned in mV

# Plotting starts here
if SeparateChannels:
	NRows = 2
	NCols = 3
else:
	NRows = 1
	NCols = 1

gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
fig = plt.figure(figsize=(6*NCols, 5*NRows))
fig.suptitle('Background RMS', fontsize='x-large') 

# print(f'len(rms) = {len(rms)}')
# print(f'rms = {rms}')

if RMSFileGiven:
	if SeparateChannels:
		for ich in range(len(channels)):		
			ax = fig.add_subplot(gs[ich])
			ax.hist(channels_RMS[ich], bins=int(np.sqrt(len(channels_RMS[ich]))))
			ax.set_xlabel(r"Amp (mV)", fontsize='large')
	else:
		ax = fig.add_subplot(gs[0])
		ax.hist(rms, bins=int(np.sqrt(len(rms))))
		ax.set_xlabel(r"Amp (mV)", fontsize='large')
else:

	if SeparateChannels:
		for ich in range(len(channels)):
			ax = fig.add_subplot(gs[ich])
			ax.hist(RMSList[ich], bins=int(np.sqrt(len(RMSList[ich]))), label=channels[ich])
			ax.set_xlabel(r"Amp (mV)", fontsize='large')
			ax.legend(loc='best', prop={'size': 12})
	else:
		ax = fig.add_subplot(gs[0])
		ax.hist(RMSList[0], bins=int(np.sqrt(len(RMSList[0]))))
		ax.set_xlabel(r"Amp (mV)", fontsize='large')

fig.savefig(path + f"/{elec_resp}RMS_{dataset}", bbox_inches='tight')