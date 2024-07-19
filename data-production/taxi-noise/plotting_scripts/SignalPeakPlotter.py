import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from icecube.icetray import I3Units
from scipy.signal import hilbert

ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
dataset = 'Auger_v2'
# dataset = 'v23'
path = ABS_PATH_HERE + f"/../data/Dataset_{dataset}"
channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]

NRows = 2
NCols = 3
gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
fig = plt.figure(figsize=(5*NCols, 4*NRows))
ax_idx = 0

# Load in channel data
for channel in channels:
    print("\nChannel: " + channel)

    SigPeaks = []
    PureSignal = np.load(f"{path}/{channel}_Signals.npy")
    for i, trace in enumerate(PureSignal):
        SigPeaks.append(np.max(np.abs(hilbert(trace)))) # Can also use abs value instead of hilbert 
        if np.max(np.abs(hilbert(trace))) / I3Units.mV > 10:
            print(i)

    # print((np.array(SigPeaks) / I3Units.mV) > 10.)

    ax = fig.add_subplot(gs[ax_idx])
    ax_idx += 1

    x = np.array(SigPeaks) / I3Units.mV
    # print(f'x={x}')
    # print(f'len(x) = {len(x)}')
    ax.hist(x, bins=int(np.sqrt(len(x))), label="Peak")
    ax.set_title(channel)
    ax.set_xlabel("Amp [mV]", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=9)

fig.suptitle(f"Signal Peaks ({dataset})")
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(path + "/SignalPeakPlot_" + dataset + ".pdf", bbox_inches='tight')
