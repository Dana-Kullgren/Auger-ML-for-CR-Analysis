import numpy as np
import matplotlib
import os

# Change this to the path for whatever file I need to access
dataset = 'Auger_v5'
path=f"/home/danakull/work/WaveformML/TrainingAndTesting/data-production/taxi-noise/data/Dataset_{dataset}"

channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]
ants = ["ant1","ant2","ant3"]
combineToAnts = True
combineAll = not combineToAnts

s = []
n = []
ns = []

if combineAll:
  
  noisy_train = []
  noisy_test = []
  pure_train = []
  pure_test = []
  snrNoisy_train = []
  snrNoisy_test = []

  traces_train = []
  traces_test = []
  labels_train = []
  labels_test = []
  snrTraces_train = []
  snrTraces_test = []

  for ich in range(len(channels)):
    channel = channels[ich]
    print(f'Loading {channel} files')

    noisy_train.extend(np.load(path + f"/{channel}_noisy_train.npy",allow_pickle=True))
    noisy_test.extend(np.load(path + f"/{channel}_noisy_test.npy",allow_pickle=True))
    pure_train.extend(np.load(path + f"/{channel}_pure_train.npy",allow_pickle=True))
    pure_test.extend(np.load(path + f"/{channel}_pure_test.npy",allow_pickle=True))
    snrNoisy_train.extend(np.load(path + f"/{channel}_snrNoisy_train.npy",allow_pickle=True))
    snrNoisy_test.extend(np.load(path + f"/{channel}_snrNoisy_test.npy",allow_pickle=True))
    traces_train.extend(np.load(path + f"/{channel}_traces_train.npy",allow_pickle=True))
    traces_test.extend(np.load(path + f"/{channel}_traces_test.npy",allow_pickle=True))
    if ich==0:
      labels_train.extend(np.load(path + f"/{channel}_labels_train.npy",allow_pickle=True))
      labels_test.extend(np.load(path + f"/{channel}_labels_test.npy",allow_pickle=True))
    snrTraces_train.extend(np.load(path + f"/{channel}_snrTraces_train.npy",allow_pickle=True))
    snrTraces_test.extend(np.load(path + f"/{channel}_snrTraces_test.npy",allow_pickle=True))

  np.save(path + f"/noisy_train.npy", noisy_train)
  np.save(path + f"/noisy_test.npy", noisy_test)
  np.save(path + f"/pure_train.npy", pure_train)
  np.save(path + f"/pure_test.npy", pure_test)
  np.save(path + f"/snrNoisy_train.npy", snrNoisy_train)
  np.save(path + f"/snrNoisy_test.npy", snrNoisy_test)
  np.save(path + f"/traces_train.npy", traces_train)
  np.save(path + f"/traces_test.npy", traces_test)
  np.save(path + f"/labels_train.npy", labels_train)
  np.save(path + f"/labels_test.npy", labels_test)
  np.save(path + f"/snrTraces_train.npy", snrTraces_train)
  np.save(path + f"/snrTraces_test.npy", snrTraces_test)

  print(f'np.shape(noisy_train)={np.shape(noisy_train)}')
  print(f'np.shape(noisy_train)={np.shape(noisy_test)}')
  print(f'np.shape(noisy_train)={np.shape(pure_train)}')
  print(f'np.shape(noisy_train)={np.shape(pure_test)}')
  print(f'np.shape(noisy_train)={np.shape(traces_train)}')
  print(f'np.shape(noisy_train)={np.shape(traces_test)}')
  print(f'np.shape(noisy_train)={np.shape(labels_train)}')
  print(f'np.shape(noisy_train)={np.shape(labels_test)}')

if combineToAnts:

  for i in range(len(ants)):
    ant = ants[i]
    print(f'ant={ant}')
  
    noisy_train = np.load(path + f"/{ant}ch0_noisy_train.npy", allow_pickle=True)
    noisy_test = np.load(path + f"/{ant}ch0_noisy_test.npy", allow_pickle=True)
    pure_train = np.load(path + f"/{ant}ch0_pure_train.npy", allow_pickle=True)
    pure_test = np.load(path + f"/{ant}ch0_pure_test.npy", allow_pickle=True)
    snrNoisy_train = np.load(path + f"/{ant}ch0_snrNoisy_train.npy", allow_pickle=True)
    snrNoisy_test = np.load(path + f"/{ant}ch0_snrNoisy_test.npy", allow_pickle=True)

    traces_train = np.load(path + f"/{ant}ch0_traces_train.npy", allow_pickle=True)
    traces_test = np.load(path + f"/{ant}ch0_traces_test.npy", allow_pickle=True)
    labels_train = np.load(path + f"/{ant}ch0_labels_train.npy", allow_pickle=True)
    labels_test = np.load(path + f"/{ant}ch0_labels_test.npy", allow_pickle=True)
    snrTraces_train = np.load(path + f"/{ant}ch0_snrTraces_train.npy", allow_pickle=True)
    snrTraces_test = np.load(path + f"/{ant}ch0_snrTraces_test.npy", allow_pickle=True)

    print(f'np.shape(noisy_train)={np.shape(noisy_train)}')
    print(f'np.shape(noisy_test)={np.shape(noisy_test)}')

    noisy_train = np.concatenate((noisy_train[:,:,np.newaxis], np.load(path + f"/{ant}ch1_noisy_train.npy", allow_pickle=True)[:,:,np.newaxis]), axis=2)
    noisy_test = np.concatenate((noisy_test[:,:,np.newaxis], np.load(path + f"/{ant}ch1_noisy_test.npy", allow_pickle=True)[:,:,np.newaxis]), axis=2)
    pure_train = np.concatenate((pure_train[:,:,np.newaxis], np.load(path + f"/{ant}ch1_pure_train.npy", allow_pickle=True)[:,:,np.newaxis]), axis=2)
    pure_test = np.concatenate((pure_test[:,:,np.newaxis], np.load(path + f"/{ant}ch1_pure_test.npy", allow_pickle=True)[:,:,np.newaxis]), axis=2)
    snrNoisy_train = np.concatenate((snrNoisy_train[:,np.newaxis], np.load(path + f"/{ant}ch1_snrNoisy_train.npy", allow_pickle=True)[:,np.newaxis]), axis=1)
    snrNoisy_test = np.concatenate((snrNoisy_test[:,np.newaxis], np.load(path + f"/{ant}ch1_snrNoisy_test.npy", allow_pickle=True)[:,np.newaxis]), axis=1)

    traces_train = np.concatenate((traces_train[:,:,np.newaxis], np.load(path + f"/{ant}ch1_traces_train.npy", allow_pickle=True)[:,:,np.newaxis]), axis=2)
    traces_test = np.concatenate((traces_test[:,:,np.newaxis], np.load(path + f"/{ant}ch1_traces_test.npy", allow_pickle=True)[:,:,np.newaxis]), axis=2)
    snrTraces_train = np.concatenate((snrTraces_train[:,np.newaxis], np.load(path + f"/{ant}ch1_snrTraces_train.npy", allow_pickle=True)[:,np.newaxis]), axis=1)
    snrTraces_test = np.concatenate((snrTraces_test[:,np.newaxis], np.load(path + f"/{ant}ch1_snrTraces_test.npy", allow_pickle=True)[:,np.newaxis]), axis=1)

    print(f'np.shape(noisy_train)={np.shape(noisy_train)}')
    print(f'np.shape(noisy_test)={np.shape(noisy_test)}')
    print(f'np.shape(traces_train)={np.shape(traces_train)}')
    print(f'np.shape(traces_test)={np.shape(traces_test)}')

    np.save(path + f"/{ant}_noisy_train.npy", noisy_train)
    np.save(path + f"/{ant}_noisy_test.npy", noisy_test)
    np.save(path + f"/{ant}_pure_train.npy", pure_train)
    np.save(path + f"/{ant}_pure_test.npy", pure_test)
    np.save(path + f"/{ant}_snrNoisy_train.npy", snrNoisy_train)
    np.save(path + f"/{ant}_snrNoisy_test.npy", snrNoisy_test)
    np.save(path + f"/{ant}_traces_train.npy", traces_train)
    np.save(path + f"/{ant}_traces_test.npy", traces_test)
    np.save(path + f"/{ant}_labels_train.npy", labels_train)
    np.save(path + f"/{ant}_labels_test.npy", labels_test)
    np.save(path + f"/{ant}_snrTraces_train.npy", snrTraces_train)
    np.save(path + f"/{ant}_snrTraces_test.npy", snrTraces_test)