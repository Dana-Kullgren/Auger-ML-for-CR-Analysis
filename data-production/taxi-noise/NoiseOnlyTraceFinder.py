'''
This script takes data which is prepared for testing on the Classifier (ex: traces_train.py) and produces files for NoiseOnly and SigPlusNoise.
'''

import numpy as np
ants = ['ant1', 'ant2', 'ant3']
dataset = 'v22'
labels_has_one_channel = False
ABS_PATH_HERE=f"/mnt/data/danakull/WaveformML/TrainingAndTesting/data-production/taxi-noise/data/Dataset_{dataset}/"

for ant in ants:
	print(ant)
	labels_train = np.load(ABS_PATH_HERE + f'{ant}_Labels_train.npy')
	traces_train = np.load(ABS_PATH_HERE + f'{ant}_Traces_train.npy')
	labels_test = np.load(ABS_PATH_HERE + f'{ant}_Labels_test.npy')
	traces_test = np.load(ABS_PATH_HERE + f'{ant}_Traces_test.npy')
	print('Files loaded')

	if labels_test.size < 2:
		labels_has_one_channel = True
	print(np.shape(traces_train))
	print(np.shape(labels_train))
	
	if labels_has_one_channel:
		## if label file has ONE channel
		NoiseOnly_train = [traces_train[:,:,0][labels_train == 0], traces_train[:,:,1][labels_train == 0]]
		SigPlusNoise_train = [traces_train[:,:,0][labels_train == 1], traces_train[:,:,1][labels_train == 1]]
		print('Training files complete')
		NoiseOnly_test = [traces_test[:,:,0][labels_test == 0], traces_test[:,:,1][labels_test == 0]]
		SigPlusNoise_test = [traces_test[:,:,0][labels_test == 1], traces_test[:,:,1][labels_test == 1]]
		print('Testing files complete')

		## Duplicate labels so it has two identical channels (this is the required file type for the Classifier)
		labels_train = np.reshape(labels_train, (labels_train.shape[0], 1))
		labels_test = np.reshape(labels_test, (labels_test.shape[0], 1))
		labels_train = np.append(labels_train, labels_train, axis=1)
		labels_test = np.append(labels_test, labels_test, axis=1)
		print(f'labels_train = {np.shape(labels_train)}')
		print(f'labels_test = {np.shape(labels_test)}')
		np.save(ABS_PATH_HERE + f'{ant}_Labels_train.npy', labels_train)
		np.save(ABS_PATH_HERE + f'{ant}_Labels_test.npy', labels_test)

	else:
		## if label file has TWO channels
		NoiseOnly_train = [traces_train[:,:,0][labels_train[:,0] == 0], traces_train[:,:,1][labels_train[:,1] == 0]]
		SigPlusNoise_train = [traces_train[:,:,0][labels_train[:,0] == 1], traces_train[:,:,1][labels_train[:,1] == 1]]
		print('Training files complete')
		NoiseOnly_test = [traces_test[:,:,0][labels_test[:,0] == 0], traces_test[:,:,1][labels_test[:,1] == 0]]
		SigPlusNoise_test = [traces_test[:,:,0][labels_test[:,0] == 1], traces_test[:,:,1][labels_test[:,1] == 1]]
		print('Testing files complete')

	print(np.shape(NoiseOnly_train))
	print(np.shape(SigPlusNoise_train))
	print(np.shape(NoiseOnly_test))
	print(np.shape(SigPlusNoise_test))

	np.save(ABS_PATH_HERE + f'{ant}_NoiseOnly_train.npy', NoiseOnly_train)
	np.save(ABS_PATH_HERE + f'{ant}_SigPlusNoise_train.npy', SigPlusNoise_train)
	np.save(ABS_PATH_HERE + f'{ant}_NoiseOnly_test.npy', NoiseOnly_test)
	np.save(ABS_PATH_HERE + f'{ant}_SigPlusNoise_test.npy', SigPlusNoise_test)
