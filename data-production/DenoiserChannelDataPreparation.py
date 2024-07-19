import numpy as np
from sklearn.model_selection import train_test_split
import os

dataset = 'Auger_v5'
DataDir=f"/home/danakull/work/WaveformML/TrainingAndTesting/data-production/taxi-noise/data/Dataset_{dataset}"
channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]
antennas = ["ant1","ant2","ant3"]
CombineChannels = False
CombineAllChannels = False

snrNoisy = [[] for i in range(len(channels))]
NoisySignal = [[] for i in range(len(channels))]
PureSignal = [[] for i in range(len(channels))]

for ich in range(len(channels)):
	channel = channels[ich]
	print(channel)
	# Load channel data
	NoisySignal[ich] = np.load(DataDir + f"/{channel}_SigPlusNoise.npy",allow_pickle=True)
	PureSignal[ich] = np.load(DataDir + f"/{channel}_Signals.npy",allow_pickle=True)
	snrNoisy[ich] = np.load(DataDir + f"/SNRNoisy_{channel}.npy",allow_pickle=True)

if CombineChannels:
	print(np.shape(NoisySignal))
	print(np.shape(snrNoisy))
	if CombineAllChannels:
		SNRs = np.transpose(snrNoisy, (1,0))
		input_data = np.transpose(NoisySignal, (1,2,0))
		output_data = np.transpose(PureSignal, (1,2,0))
		# for i in NoisySignal:
		# 	for j in range(len(NoisySignal[0])):
				# SNRs[j][i] = snrNoisy[i][j]
				# for k in range(len(NoisySignal[0][0])):
					# input_data[j][k][i] = NoisySignal[i][j][k]
					# output_data[j][k][i] = PureSignal[i][j][k]
		# input_list = []
		# output_list = []
		# SNR_list = []

		# for j in range(len(NoisySignal[0])):
		# 	print(j={j})
		# 	input_list.append(NoisySignal[j][:, :, np.newaxis])
		# 	output_list.append(PureSignal[j][:, :, np.newaxis])
		# 	SNR_list.append(snrNoisy[j][:, np.newaxis])

		# input_data = np.concatenate([input_list], axis=-1)
		# output_data = np.concatenate([output_list], axis=-1)
		# SNRs = np.concatenate([SNR_list], axis=-1)

		print(np.shape(input_data))

		noisy_train, noisy_test, pure_train, pure_test, snrNoisy_train, snrNoisy_test = train_test_split(input_data,
							                                 output_data, SNRs,
							                                 random_state=42,
							                                 test_size=0.2)

		np.save(DataDir + "/_6ChCombined_noisy_train.npy", noisy_train)
		np.save(DataDir + "/_6ChCombined_noisy_test.npy", noisy_test)
		np.save(DataDir + "/_6ChCombined_pure_train.npy", pure_train)
		np.save(DataDir + "/_6ChCombined_pure_test.npy", pure_test)
		
		np.save(DataDir + "/_6ChCombined_snrNoisy_train.npy", snrNoisy_train)
		np.save(DataDir + "/_6ChCombined_snrNoisy_test.npy", snrNoisy_test)

	else:
		for i in range(len(antennas)):
			print(f'i={i}: (2*i), (2*i)+1 = {(2*i), (2*i)+1}')
			input_data = np.concatenate([NoisySignal[2*i][:, :, np.newaxis], NoisySignal[(2*i)+1][:, :, np.newaxis]], axis=-1)
			output_data = np.concatenate([PureSignal[2*i][:, :, np.newaxis], PureSignal[(2*i)+1][:, :, np.newaxis]], axis=-1)
			SNRs = np.concatenate([snrNoisy[2*i][:, np.newaxis], snrNoisy[(2*i)+1][:, np.newaxis]], axis=-1)
			check_if_combined = '_'

		noisy_train, noisy_test, pure_train, pure_test, snrNoisy_train, snrNoisy_test = train_test_split(input_data,
                                                             output_data, SNRs,
                                                             random_state=42,
                                                             test_size=0.2)


		np.save(DataDir + f"/{antennas[i]}_noisy_train.npy", noisy_train)
		np.save(DataDir + f"/{antennas[i]}_noisy_test.npy", noisy_test)
		np.save(DataDir + f"/{antennas[i]}_pure_train.npy", pure_train)
		np.save(DataDir + f"/{antennas[i]}_pure_test.npy", pure_test)
		
		np.save(DataDir + f"/{antennas[i]}_snrNoisy_train.npy", snrNoisy_train)
		np.save(DataDir + f"/{antennas[i]}_snrNoisy_test.npy", snrNoisy_test)

else:
	for ich in range(len(channels)):
		channel = channels[ich]
		noisysignal = NoisySignal[ich]
		puresignal = PureSignal[ich]
		snrnoisy = snrNoisy[ich]

		# Create files that the Classifier can use
		noisy_train, noisy_test, pure_train, pure_test, snrNoisy_train, snrNoisy_test = train_test_split(noisysignal,
	                                                             puresignal, snrnoisy,
	                                                             random_state=42,
	                                                             test_size=0.2)

		np.save(DataDir + "/{0}_noisy_train.npy".format(channel), noisy_train)
		np.save(DataDir + "/{0}_noisy_test.npy".format(channel), noisy_test)
		np.save(DataDir + "/{0}_pure_train.npy".format(channel), pure_train)
		np.save(DataDir + "/{0}_pure_test.npy".format(channel), pure_test)
		
		np.save(DataDir + "/{0}_snrNoisy_train.npy".format(channel), snrNoisy_train)
		np.save(DataDir + "/{0}_snrNoisy_test.npy".format(channel), snrNoisy_test)