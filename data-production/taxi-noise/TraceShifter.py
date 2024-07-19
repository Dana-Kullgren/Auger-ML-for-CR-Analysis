import numpy as np
import random
import os

# Used to shift traces after the dataset has already been created

dataset = 'v21'
path = f"/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_{dataset}/"
channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]
ChannelSeparated = False


if ChannelSeparated:
	s = [[] for i in range(len(channels))]
	n = [[] for i in range(len(channels))]
	ns = [[] for i in range(len(channels))]

	# load traces
	for ich in range(len(channels)):
		ch = channels[ich]
		s[ich] = np.load(path + ch + "_Signals.npy")
		n[ich] = np.load(path + ch + "_NoiseOnly.npy")
		ns[ich] = np.load(path + ch + "_SigPlusNoise.npy")
		print(f'{ch} loaded')

	print(f'{channels[0]}_Signals.npy')
	print(f'np.shape(s[0])={np.shape(s[0])}')

else:
	s = np.load(path + "Signals.npy")
	n = np.load(path + "NoiseOnly.npy")
	ns = np.load(path + "SigPlusNoise.npy")

	print(f'np.shape(s)={np.shape(s)}')


# calculate number of bins to shift by
if ChannelSeparated:
	first_trace = s[0]
else:
	first_trace = s

shifInd = []
for i in range(3*len(first_trace)):
	spread = int(len(first_trace[0])*.25) # else keep the previous spread so both channels are shifted by the same amount
	shift = random.randint(-spread,spread) # shifting by random number of bins
	shifInd.append(shift)
print(f'spread={spread}')

print(f'len(shifInd) = {len(shifInd)}')

if ChannelSeparated:
	shifi = 0
	for ich in [0,2,4]:
		for i in range(len(s[ich])):
			# print(f'counter={counter}, shifInd={shifInd}')
			shifVal = shifInd[shifi]
			# print(f'shifVal={shifVal}')

			s0 = s[ich][i]
			n0 = n[ich][i]
			ns0 = ns[ich][i]

			s1 = s[ich+1][i]
			n1 = n[ich+1][i]
			ns1 = ns[ich+1][i]

			# print(f'np.shape(s0) = {np.shape(s0)}')
			# print(f'pre-roll s0[0] = {s0[0]}')

			s0 = np.roll(s0, shifVal)
			n0 = np.roll(n0, shifVal)
			ns0 = np.roll(ns0, shifVal)

			s1 = np.roll(s1, shifVal)
			n1 = np.roll(n1, shifVal)
			ns1 = np.roll(ns1, shifVal)

			# print(f'post-roll s0[0] = {s0[0]}')

			s[ich][i] = s0
			n[ich][i] = n0
			ns[ich][i] = ns0

			s[ich+1][i] = s1
			n[ich+1][i] = n1
			ns[ich+1][i] = ns1		

			# np.roll(s[ich][i], shifInd[shifi])
			# np.roll(n[ich][i], shifInd[shifi])
			# np.roll(ns[ich][i], shifInd[shifi])

			# np.roll(s[ich+1][i], shifInd[shifi])
			# np.roll(n[ich+1][i], shifInd[shifi])
			# np.roll(ns[ich+1][i], shifInd[shifi])

			shifi += 1

	print(f'shifi = {shifi}')

	print(f'Shifted_{channels[0]}_Signals.npy')
	print(f'np.shape(s[0])={np.shape(s[0])}')

	for ich in range(len(channels)):
		ch = channels[ich]
		np.save(f"{path}Shifted_{ch}_Signals.npy", s[ich])
		np.save(f"{path}Shifted_{ch}_NoiseOnly.npy", n[ich])
		np.save(f"{path}Shifted_{ch}_SigPlusNoise.npy", ns[ich])

else:
	for i in range(len(s)):
		# print(f'counter={counter}, shifInd={shifInd}')
		shifVal = shifInd[i]

		s[i] = np.roll(s[i], shifVal)
		n[i] = np.roll(n[i], shifVal)
		ns[i] = np.roll(ns[i], shifVal)

	print(f'np.shape(s)={np.shape(s)}')

	for ich in range(len(channels)):
		ch = channels[ich]
		np.save(f"{path}Shifted_Signals.npy", s[ich])
		np.save(f"{path}Shifted_NoiseOnly.npy", n[ich])
		np.save(f"{path}Shifted_SigPlusNoise.npy", ns[ich])

