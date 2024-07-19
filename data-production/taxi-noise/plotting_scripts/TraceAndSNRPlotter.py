import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from icecube.icetray import I3Units
import os

# Change this to the path for whatever file I need to access
ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
dataset = v6
path =  ABS_PATH_HERE + f"/../data/Dataset_{dataset}"

snrSig = np.load(path + "/SNRSignals.npy")
print("Signals SNR loaded")
snrNoise = np.load(path + "/SNRNoiseOnly.npy")
print("Noise SNR loaded")
snrSigPlusNoise = np.load(path + "/SNRSigPlusNoise.npy")
print("Noisy signals SNR loaded \n")

s = np.load(path + "/AllSignals.npy")
print("Signals loaded")
n = np.load(path + "/AllNoiseOnly.npy")
print("Noise loaded")
ns = np.load(path + "/AllSigPlusNoise.npy")
print("Noisy signals loaded \n")

print("Number of traces in the files:")
print("Signal", len(s))
print("Noise Only", len(n))
print("Signal Plus Noise", len(ns))

# Plotting starts here

# 1000 5000 10,000 50,000 100,000
idxs = [254069, 253986, 254020, 253847, 253969]
for idx in idxs:
    NRows = 2
    NCols = 3
    gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
    fig = plt.figure(figsize=(6*NCols, 5*NRows))
    plotindex = idx # this is index of the trace I am plotting from the file
    #fig.suptitle('SNR Distributions and Traces for Pure Signals, Noise, and Noisy Signals')  

    ax = fig.add_subplot(gs[0])
    selection = (snrSig != 0)
    ax.hist(np.log10(snrSig[selection]), bins=int(np.sqrt(len(snrSig[selection]))), label="Pure Signal")
    ax.set_xlabel(r"log$_{10}$(Signal SNR)")
    ax.legend(loc='best', prop={'size': 8})
   
    ax = fig.add_subplot(gs[1])
    selection = (snrNoise != 0)
    ax.hist(np.log10(snrNoise[selection]), bins=int(np.sqrt(len(snrNoise[selection]))), label="Noise")
    ax.set_xlabel(r"log$_{10}$(Noise SNR)")
    ax.legend(loc='best', prop={'size': 8})

    ax = fig.add_subplot(gs[2])
    selection = (snrSigPlusNoise != 0)
    ax.hist(np.log10(snrSigPlusNoise[selection]), bins=int(np.sqrt(len(snrSigPlusNoise[selection]))), label="Signal Plus Noise")
    ax.set_xlabel(r"log$_{10}$(Signal Plus Noise SNR)")
    ax.legend(loc='best', prop={'size': 8})

    ax = fig.add_subplot(gs[3])
    ax.set_title("SNR Value: {0:0.1f}".format(snrSig[plotindex]))
    x1 = np.array(s[plotindex]) / I3Units.mV
    ax.plot(x1, label = "Pure Signal")
    ax.set_xlim(1024,3072)                      # Delete to show full domain
    ax.set_ylim(-1.1*max(x1), 1.1*max(x1))
    ax.set_xlabel("Time Bins")
    ax.set_ylabel(r"Amplitude (mV)")
    ax.legend(loc='best', prop={'size': 8})
    
    ax = fig.add_subplot(gs[4])
    ax.set_title("SNR Value: {0:0.1f}".format(snrNoise[plotindex]))
    x3 = np.array(n[plotindex]) / I3Units.mV
    ax.plot(x3, label = "Noise")
    ax.set_xlim(1024,3072)                      # Delete to show full domain
    ax.set_ylim(-1.1*max(x3), 1.1*max(x3))
    ax.set_xlabel("Time Bins")
    ax.set_ylabel(r"Amplitude (mV)")
    ax.legend(loc='best', prop={'size': 8})

    ax = fig.add_subplot(gs[5])
    # plt.title("SNR Value: " + str(snrSigPlusNoise[plotindex]))
    ax.set_title("SNR Value: {0:0.1f}".format(snrSigPlusNoise[plotindex]))
    x2 = np.array(ns[plotindex]) / I3Units.mV
    ax.plot(x2, label = "Signal Plus Noise")
    ax.set_xlim(1024,3072)                      # Delete to show full domain
    ax.set_ylim(-1.1*max(x2), 1.1*max(x2))
    ax.set_xlabel("Time Bins")
    ax.set_ylabel(r"Amplitude (mV)")
    ax.legend(loc='best', prop={'size': 8})

    fig.savefig(path + f"/SNR_Plots_{dataset}/SNR_Distribution_v6_plotindex=" + str(idx) + ".pdf", bbox_inches='tight')
    print("Trace " + str(idx) + " plotted")
