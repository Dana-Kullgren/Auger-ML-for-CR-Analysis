import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statistics as stat
import os

# change this to the path for whatever file I need to access
ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
dataset = 'Auger_v3'
path = ABS_PATH_HERE + f"/../data/Dataset_{dataset}"

# Update these paths
snrSig = np.load(path + f"/SNR_Signals.npy", allow_pickle=True)
print("Signals SNR loaded")
print(f'snrSig={snrSig}')
print(f'np.shape(snrSig)={np.shape(snrSig)}')
snrNoise = np.load(path + f"/SNR_NoiseOnly.npy", allow_pickle=True)
print("Noise SNR loaded")
print(f'np.shape(snrNoise)={np.shape(snrNoise)}')
snrSigPlusNoise = np.load(path + f"/SNR_SigPlusNoise.npy", allow_pickle=True)
print("Noisy signals SNR loaded")
print(f'np.shape(snrSigPlusNoise)={np.shape(snrSigPlusNoise)}')

print("Pure signals mean SNR: ", stat.mean(snrSig))
print("Noise Only mean SNR: ", stat.mean(snrNoise))
print("Sig Plus Noise mean SNR: ", stat.mean(snrSigPlusNoise))

# Plotting starts here
NRows = 1
NCols = 3
gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
fig = plt.figure(figsize=(6*NCols, 5*NRows))
fig.suptitle('SNR Distributions for Pure Signals, Noise, and Noisy Signals', fontsize='x-large') 

ax = fig.add_subplot(gs[0])
snrSig = snrSig[~np.isnan(snrSig)]
selection = (snrSig != 0)
print(f'bins={int(np.sqrt(len(snrSig[selection])))}')
ax.hist(np.log10(snrSig[selection]), bins=int(np.sqrt(len(snrSig[selection]))), label="Pure Signal")
ax.set_yscale('log')
ax.set_xlabel(r"log$_{10}$(Signal SNR)", fontsize='large')
ax.legend(loc='best', prop={'size': 12})

ax = fig.add_subplot(gs[1])
selection = (snrNoise != 0)
ax.hist(np.log10(snrNoise[selection]), bins=int(np.sqrt(len(snrNoise[selection]))), label="Noise")
ax.set_yscale('log')
ax.set_xlabel(r"log$_{10}$(Noise SNR)", fontsize='large')
ax.legend(loc='best', prop={'size': 12})

ax = fig.add_subplot(gs[2])
selection = (snrSigPlusNoise != 0)
ax.hist(np.log10(snrSigPlusNoise[selection]), bins=int(np.sqrt(len(snrSigPlusNoise[selection]))), label="Signal Plus Noise")
ax.set_yscale('log')
ax.set_xlabel(r"log$_{10}$(Signal Plus Noise SNR)", fontsize='large')
ax.legend(loc='best', prop={'size': 12})

fig.savefig(path + f"/Only_SNR_Distribution_Small_{dataset}", bbox_inches='tight')