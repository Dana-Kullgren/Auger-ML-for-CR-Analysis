import numpy as np
import matplotlib
import os

# Please note: I am only using Run 0-7 because Runs 8 and 9 aren't indexing correctly
# New note: I am now only running Run 1-3 because the file sizes are too large when all runs are combined

# Change this to the path for whatever file I need to access
dataset = 'Auger_v3'
path = f"/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_{dataset}"

######################################################################################

# To consolidate data that is NOT divided by channel, uncomment this section

# s = np.load(path + "/Run0_Signals.npy")
# n = np.load(path + "/Run0_NoiseOnly.npy")
# ns = np.load(path + "/Run0_SigPlusNoise.npy")

# # print(np.shape(s))
# # print(np.shape(n))
# # print(np.shape(ns))

# for i in range(1, 10):
#     s = np.vstack((s, np.load(path + "/Run" + str(i) + "_Signals.npy")))
#     n = np.vstack((n, np.load(path + "/Run" + str(i) + "_NoiseOnly.npy")))
#     ns = np.vstack((ns, np.load(path + "/Run" + str(i) + "_SigPlusNoise.npy")))

# np.save(path + "/AllSignals.npy", s)
# np.save(path + "/AllNoiseOnly.npy", n)
# np.save(path + "/AllSigPlusNoise.npy", ns)

######################################################################################

# To consolidate data that IS divided by channel, uncomment this section

Small = False
VerySmall = False
channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]

TotalSigTraces = 0
TotalBackground = 0
SigTraces = [0,0,0,0,0,0]
NoiseTraces = [0,0,0,0,0,0]

first_idx = 0
# first_idx = 1
# first_idx = 2

for ich in range(len(channels)):
    channel = channels[ich]

    s = np.load(path + f"/Run{first_idx}_{channel}_Signals.npy", allow_pickle=True)
    n = np.load(path + f"/Run{first_idx}_{channel}_NoiseOnly.npy", allow_pickle=True)
    ns = np.load(path + f"/Run{first_idx}_{channel}_SigPlusNoise.npy", allow_pickle=True)

    print(channel + ":")
    print(f"Run {first_idx}")
    print("Signals: ", np.shape(s))
    print("NoiseOnly: ", np.shape(n))
    print("SigPlusNoise: ", np.shape(ns))

    if Small:
        end_range = 4
    elif VerySmall:
        end_range = 2
    else:
        end_range=10
    # for i in range(1, end_range):
    # for i in [2,4,6,7,8,9]:
    for i in range(1, end_range):
        s1 = np.load(path + "/Run{0}_{1}_Signals.npy".format(str(i), channel), allow_pickle=True)
        n1 = np.load(path + "/Run{0}_{1}_NoiseOnly.npy".format(str(i), channel), allow_pickle=True)
        ns1 = np.load(path + "/Run{0}_{1}_SigPlusNoise.npy".format(str(i), channel), allow_pickle=True)

        print(f"Run {i}")
        print("Signals: ", np.shape(s))
        print("NoiseOnly: ", np.shape(n))
        print("SigPlusNoise: ", np.shape(ns))

        s = np.vstack((s, s1))
        n = np.vstack((n, n1))
        ns = np.vstack((ns, ns1))

    print('Combined runs')
    print("Signals: ", np.shape(s))
    print("NoiseOnly: ", np.shape(n))
    print("SigPlusNoise: ", np.shape(ns))

    for trace in s:
        if np.max(trace) == 0:
            NoiseTraces[ich] += 1
        else:
            SigTraces[ich] += 1

    if Small:
        tag = "Small"
    elif VerySmall:
        tag = "VerySmall"
    else:
        tag=""

    np.save(path + f"/{channel}_{tag}Signals.npy", s)
    np.save(path + f"/{channel}_{tag}NoiseOnly.npy", n)
    np.save(path + f"/{channel}_{tag}SigPlusNoise.npy", ns)

TotalSigTraces = sum(SigTraces)
print("Total number of signal traces:", TotalSigTraces)
TotalBackground = sum(NoiseTraces)
print("Total number of background traces:", TotalBackground)
######################################################################################
