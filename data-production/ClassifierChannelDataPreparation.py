import numpy as np
from sklearn.model_selection import train_test_split	
import os

dataset = 'Auger_v5'
# Dana's
DataDir=f"/home/danakull/work/WaveformML/TrainingAndTesting/data-production/taxi-noise/data/Dataset_{dataset}"
# Paula's
# DataDir=f"/home/paulagm/TrainingAndTesting/data-production/taxi-noise/data/Dataset_{dataset}"
channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]
antennas = ["ant1","ant2","ant3"]
CombineChannels = False

snrTraces = [[] for i in range(len(channels))]
SigNoise = [[] for i in range(len(channels))]
Label = [[] for i in range(len(channels))]

for ich in range(len(channels)):

	channel = channels[ich]
	# Load channel data
	Signal = np.load(DataDir + f"/{channel}_SigPlusNoise.npy",allow_pickle=True)
	Background = np.load(DataDir + f"/{channel}_NoiseOnly.npy",allow_pickle=True)
	snrTraces[ich] = np.load(DataDir + f"/SNRTraces_{channel}.npy",allow_pickle=True)

	# Concatenate signal and background traces
	SigNoise[ich] = np.concatenate((Signal, Background))

	# Create list of labels
	LabelS = np.ones(len(Signal))
	LabelN = np.zeros(len(Background))
	Label[ich] = np.concatenate((LabelS, LabelN))
	print(f'{channel}: Label[ich]={Label[ich]}')
print(f'Label={Label}')

print(f'np.shape(SigNoise[0])={np.shape(SigNoise[0])}')

if CombineChannels:
	for i in range(len(antennas)):
		print(f'i={i}: (2*i), (2*i)+1 = {(2*i), (2*i)+1}')
		input_data = np.concatenate([SigNoise[2*i][:, :, np.newaxis], SigNoise[(2*i)+1][:, :, np.newaxis]], axis=-1)
		labels = np.concatenate([Label[2*i][:, np.newaxis], Label[(2*i)+1][:, np.newaxis]], axis=-1)
		SNRs = np.concatenate([snrTraces[2*i][:, np.newaxis], snrTraces[(2*i)+1][:, np.newaxis]], axis=-1)

		# Create files that the Classifier can use
		traces_train, traces_test, labels_train, labels_test, snrTraces_train, snrTraces_test = train_test_split(input_data,
                                                         labels, SNRs,
                                                         random_state=42,
                                                         test_size=0.2)
		
		print("traces_train: ", np.shape(traces_train))
		print("traces_test: ", np.shape(traces_test))

		np.save(DataDir + "/{0}_traces_train.npy".format(antennas[i]), traces_train)
		np.save(DataDir + "/{0}_traces_test.npy".format(antennas[i]), traces_test)
		np.save(DataDir + "/{0}_labels_train.npy".format(antennas[i]), labels_train)
		np.save(DataDir + "/{0}_labels_test.npy".format(antennas[i]), labels_test)
		
		np.save(DataDir + "/{0}_snrTraces_train.npy".format(antennas[i]), snrTraces_train)
		np.save(DataDir + "/{0}_snrTraces_test.npy".format(antennas[i]), snrTraces_test)

else:
	for ich in range(len(channels)):
		channel = channels[ich]
		signoise = SigNoise[ich]
		label = Label[ich]
		snrtraces = snrTraces[ich]

		print(f'channel={channel}')
		# print(f'SigNoise[:10]={SigNoise[:10]}')
		print(f'Label={Label}')
		print(f'snrTraces={snrTraces}')

		# Create files that the Classifier can use
		traces_train, traces_test, labels_train, labels_test, snrTraces_train, snrTraces_test = train_test_split(signoise,
	                                                         label, snrtraces,
	                                                         random_state=42,
	                                                         test_size=0.2)
		
		print("traces_train: ", np.shape(traces_train))
		print("traces_test: ", np.shape(traces_test))

		np.save(DataDir + "/{0}_traces_train.npy".format(channel), traces_train)
		np.save(DataDir + "/{0}_traces_test.npy".format(channel), traces_test)
		np.save(DataDir + "/{0}_labels_train.npy".format(channel), labels_train)
		np.save(DataDir + "/{0}_labels_test.npy".format(channel), labels_test)
		
		np.save(DataDir + "/{0}_snrTraces_train.npy".format(channel), snrTraces_train)
		np.save(DataDir + "/{0}_snrTraces_test.npy".format(channel), snrTraces_test)
