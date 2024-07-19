import numpy as np
import os
from I3Tray import I3Tray
from icecube import icetray, radcube, dataclasses, taxi_reader
from icecube.radcube import defaults
from icecube.icetray import I3Units
import random
#from icecube.icetray.i3logging import log_info, log_error, log_fatal
from icecube.icetray.i3logging import log_info, log_error, log_fatal, log_trace, log_warn #added log_warn
icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)

class DataExtractor(icetray.I3Module):
 
  def __init__(self,ctx):
    icetray.I3Module.__init__(self,ctx)
    self.inputNameNoisy = ""
    self.AddParameter("InputNameNoisy", "Input filtered cleaned noisy voltages", self.inputNameNoisy)
    self.inputNameNoise = ""
    self.AddParameter("InputNameNoise", "Input filtered cleaned noise voltages", self.inputNameNoise)
    self.inputNameTrue = ""
    self.AddParameter("InputNameTrue", "Input filtered scaled signal", self.inputNameTrue)
    self.arguments = ""
    self.AddParameter("Arguments", "Arguments passed to MakeTraces", self.arguments)
    self.dataset = ""
    self.AddParameter("Dataset", "Dataset version (ex: v11)", self.dataset)
    self.RMSCutoff = ""
    # self.AddParameter("RMSCutoff", "Cutoff value for RMS", self.RMSCutoff)
    # self.MakeRMSCutoff = ""
    # self.AddParameter("MakeRMSCutoff", "Determines if RMS cutoff is needed", self.MakeRMSCutoff)

  def Configure(self):
    self.inputNameNoisy = self.GetParameter("InputNameNoisy")
    self.inputNameNoise = self.GetParameter("InputNameNoise")
    self.inputNameTrue = self.GetParameter("InputNameTrue")
    self.arguments = self.GetParameter("Arguments")
    self.dataset = self.GetParameter("Dataset")
    # self.RMSCutoff = self.GetParameter("RMSCutoff")
    # self.MakeRMSCutoff = self.GetParameter("MakeRMSCutoff")
    self.Signals = [[] for i in range(3*2)]
    self.SigPlusNoise  = [[] for i in range(3*2)]
    self.NoiseOnly = [[] for i in range(3*2)]
    # self.SNRSignals = [[] for i in range(3*2)]
    # self.SNRSigPlusNoise  = [[] for i in range(3*2)]
    # self.SNRNoiseOnly = [[] for i in range(3*2)]
    self.signalcounter = 0
    self.noiseonlycounter = 0
    self.sigplusnoisecounter = 0
    # self.RMSList = [[] for i in range(6)]
    # self.ROIList = [[] for i in range(3*2)]
    # self.TraceLengthList = [[] for i in range(3*2)]

    # # Added to save SNR data
    # self.bump2Traces = [[] for i in range(6)]      # There is one sublist for each channel
    # self.bump2TraceLength = [[] for i in range(6)] # This will save the readout mode
    # self.bump2TraceROI = [[] for i in range(6)]    # This will save the ROI

    # self.bump3Traces = [[] for i in range(6)]
    # self.bump3TraceLength = [[] for i in range(6)]
    # self.bump3TraceROI = [[] for i in range(6)]    

  def Physics(self, frame):
    # Signals = frame["TheFilteredScaledSignal"] # Its an antenna data map
    # SigPlusNoise = frame["TheFilteredCleanedNoisyVoltage"]
    # NoiseOnly = frame["TheFilteredCleanedNoiseVoltage"]

    # print(f'frame.keys() = {frame.keys()}')

    # if "RadioTraceLength" in frame:
    #   print("found item")

    Signals = frame[self.inputNameTrue] # Its an antenna data map
    SigPlusNoise = frame[self.inputNameNoisy]
    NoiseOnly = frame[self.inputNameNoise]
    
    print(f'len(Signals)={len(Signals)}')
    print(f'len(SigPlusNoise)={len(SigPlusNoise)}')
    print(f'len(NoiseOnly)={len(NoiseOnly)}')
    # ROI = frame["RadioAntennaROI"].value
    # TraceLength = frame["RadioTraceLength"].value

    # # def GetSNR(Trace):
    # def GetSNR(Trace, BelowRMSCut):
    #     '''
    #     Return the Signal to Noise Ratio. Signal is just the peak of the trace.
    #     Medain RMS of chunck of trace. 
    #     '''
    #     from scipy.signal import hilbert
    #     SigPeak = np.max(np.abs(hilbert(Trace))) # Can also use abs value instead of hilbert 
    #     Chunks = np.array_split(Trace, 10)  # Split the trace in to 10 small chunks
    #     ChunkRMS_squared = [(sum(chunk**2))/len(chunk) for chunk in Chunks] ## RMS^2 of each chunk
    #     RMS_Median = np.median(ChunkRMS_squared) ## Chunk with signal in it.
    #     # BelowRMSCut = True
    #     # if RMS_Median > self.RMSCutoff:
    #     #   BelowRMSCut = False
    #     return SigPeak**2/RMS_Median, SigPeak, RMS_Median

    # def GetAntData(AntMap, List, SNRList, counter, ROIMap=ROI, TraceLengthMap=TraceLength, ROIList=self.ROIList, TraceLengthList=self.TraceLengthList): ## give it a Antenna Map and empty list, it will give you list of time series
    def GetAntData(AntMap, List, counter): ## give it a Antenna Map and empty list, it will give you list of time series
      nant = 3
      nch = 2 # Under the assumption that all antennas have 2 channels
      print('GetAntData() has started running...')
      print(f'len(AntMap)={len(AntMap)}')
      print(AntMap.keys())
      for iant, antkey in enumerate(AntMap.keys()):
        print(f'antkey={antkey}')
        channelMapSig = AntMap[antkey]
        # channelROIMapSig = ROIMap[antkey]
        # channelTraceLengthMapSig = TraceLengthMap[antkey]
        for ich, key in enumerate(channelMapSig.keys()):
          print(f'key={key}')
          chdata = channelMapSig[key]
          fft = chdata.GetFFTData()
          timeseries = fft.GetTimeSeries()
          # spectrum = fft.GetFrequencySpectrum() ## This will give you frequency amplitudes (y-axis)
          # specbinning = spectrum.binning
          # freqs = [specbinning * i for i in range(spectrum.GetSize())] ## This will give you Frequencies (x-axis)
          # roi = channelROIMapSig[ich]
          # trace_length = channelTraceLengthMapSig[ich]
          timeseriespy = [timeseries[i] for i in range(timeseries.GetSize())]
          # spectrumpy = [spectrum[i] for i in range(spectrum.GetSize())]

          List[counter % 6].append(timeseriespy)
          # List[0][counter % 6].append(timeseriespy)
          # List[1][counter % 6].append(spectrumpy)
          # List[2][counter % 6].append(freqs)

          counter += 1
      return counter

      # Uncomment this if I can't the channel-divided method to work
      # for iant, antkey in enumerate(AntMap.keys()):
      #   channelMapSig = AntMap[antkey]
      #   for key in channelMapSig.keys():
      #     chdata = channelMapSig[key]
      #     fft = chdata.GetFFTData()
      #     timeseries = fft.GetTimeSeries()
      #     # spectrum = fft.GetFrequencySpectrum()
      #     timeseriespy = [timeseries[i] for i in range(timeseries.GetSize())]
      #     List.append(timeseriespy)
      # return List

    channels = ["ant1ch0","ant1ch1","ant2ch0","ant2ch1","ant3ch0","ant3ch1"]

    # def GetRMS(NoiseOnly, RMSList):
    #   # print(len(NoiseOnly))
    #   for i in range(len(channels)):        
    #     for trace in NoiseOnly[i]:
    #       # print(f'type(trace) = {type(trace)}')
    #       trace = np.array(trace)
    #       # print(np.shape(trace))
    #       peak, rms, snr = radcube.GetChunkSNR(trace, No_Chunks=10)
    #       rms = rms/I3Units.mV

    #       # print(f'peak = {peak}')
    #       # print(f'rms = {rms}')
    #       # print(f'snr = {snr}')
    #       # print(f'np.shape(RMSList) = {np.shape(RMSList)}')
    #       # print(f'np.shape(RMSList[i]) = {np.shape(RMSList[i])}')
    #       RMSList[i] = np.append(RMSList[i], rms)
    #   return RMSList

    print(f'len(Signals)={len(Signals)}')
    print(f'len(self.Signals)={len(self.Signals)}')
    
    self.signalcounter = GetAntData(Signals, self.Signals, self.signalcounter) #Should get 976 (nant*nch)
    self.sigplusnoisecounter = GetAntData(SigPlusNoise, self.SigPlusNoise, self.sigplusnoisecounter)
    self.noiseonlycounter = GetAntData(NoiseOnly, self.NoiseOnly, self.noiseonlycounter)

    print(f'--------------------------{len(self.Signals[0])} /n {len(self.Signals[1])}')
    print(f'-----------------------------------')

    print("Signal counter: ", self.signalcounter) # If these are non-zero, it's good
    print("NoiseOnly counter: ", self.noiseonlycounter)
    print("SigPlusNoise counter: ", self.sigplusnoisecounter)

    # Find RMS values for NoiseOnlyTraces
    # List=[[] for i in range(len(channels))]
    # for i in range(len(channels)):
      # self.RMSList = GetRMS(self.NoiseOnly[0], self.RMSList) # rms is returned in mV
    # print(f'self.RMSList = {self.RMSList}')
    # print(f'np.shape(self.RMSList) = {np.shape(self.RMSList)}')
    # print(f'self.RMSList[0] = {self.RMSList[0]}')
    # print(f'np.shape(self.RMSList[0]) = {np.shape(self.RMSList[0])}')
    # print(f'self.RMSList[0][0] = {self.RMSList[0][0]}')
    # print(f'np.shape(self.RMSList[0][0]) = {np.shape(self.RMSList[0][0])}')
    # print(f'self.RMSCutoff = {self.RMSCutoff}')

    # AcceptableRMS = [[] for i in range(len(channels))]
    # for i in range(len(channels)):
    #   print(f'np.shape(AcceptableRMS) = {np.shape(AcceptableRMS)}')
    #   print(f'np.shape(AcceptableRMS[i]) = {np.shape(AcceptableRMS[i])}')
    #   for rms in RMSList[i]:
    #     if rms < self.RMSCutoff:
    #       AcceptableRMS[i] = np.append(AcceptableRMS[i], 1)
    #     else:
    #       AcceptableRMS[i] = np.append(AcceptableRMS[i], 0)
    # print(f'AcceptableRMS = {AcceptableRMS}')

    # # Keep only the traces that are under the cutoff value
    # for i in range(len(channels)):
    #   self.Signals[0][i] = [self.Signals[0][i][j] for j in range(len(self.Signals)) if AcceptableRMS[i][j]==1]
    #   self.NoiseOnly[0][i] = [self.NoiseOnly[0][i][j] for j in range(len(self.Signals)) if AcceptableRMS[i][j]==1]
    #   self.SigPlusNoise[0][i] = [self.SigPlusNoise[0][i][j] for j in range(len(self.Signals)) if AcceptableRMS[i][j]==1]

    #   print(f'len(self.Signals[0][i]) after the RMS cutoff = {len(self.Signals[0][i])}')

    # Keep only the traces that are under the cutoff value
    for i in range(len(channels)):
      # print(f'len(self.RMSList[i]) = {len(self.RMSList[i])}')
      print(f'len(self.Signals) = {len(self.Signals)}')
      # print(f'self.RMSList[i][0] = {self.RMSList[i][0]}')
      # print(f'self.RMSList[i][1] = {self.RMSList[i][1]}')
      # print(f'self.RMSList[i][2] = {self.RMSList[i][2]}')
      # print(f'self.RMSList[i][3] = {self.RMSList[i][3]}')
      # print(f'self.RMSList[i][4] = {self.RMSList[i][4]}')
      # print(f'self.RMSList[i][5] = {self.RMSList[i][5]}')
      # print(f'self.RMSList[i][6] = {self.RMSList[i][6]}')
      # print(f'self.RMSList[i][7] = {self.RMSList[i][7]}')
      # print(f'self.RMSList[i][8] = {self.RMSList[i][8]}')
      # print(f'self.RMSList[i][9] = {self.RMSList[i][9]}')
      # print(f'self.RMSList[i][10] = {self.RMSList[i][10]}')

      # if self.MakeRMSCutoff:
      #   self.Signals[0][i] = [self.Signals[0][i][j] for j in range(len(self.Signals[0][i])) if self.RMSList[i][j] < self.RMSCutoff]
      #   self.NoiseOnly[0][i] = [self.NoiseOnly[0][i][j] for j in range(len(self.Signals[0][i])) if self.RMSList[i][j] < self.RMSCutoff]
      #   self.SigPlusNoise[0][i] = [self.SigPlusNoise[0][i][j] for j in range(len(self.Signals[0][i])) if self.RMSList[i][j] < self.RMSCutoff]
      #   print(f'len(self.Signals[0]) after the RMS cutoff = {len(self.Signals[0])}')
      #   print(f'len(self.Signals[0][i]) after the RMS cutoff = {len(self.Signals[0][i])}')

      # else:
      #   self.Signals[0][i] = [self.Signals[0][i][j] for j in range(len(self.Signals[0][i]))]
      #   self.NoiseOnly[0][i] = [self.NoiseOnly[0][i][j] for j in range(len(self.Signals[0][i]))]
      #   self.SigPlusNoise[0][i] = [self.SigPlusNoise[0][i][j] for j in range(len(self.Signals[0][i]))]

      self.Signals[i] = [self.Signals[i][j] for j in range(len(self.Signals[i]))]
      self.NoiseOnly[i] = [self.NoiseOnly[i][j] for j in range(len(self.Signals[i]))]
      self.SigPlusNoise[i] = [self.SigPlusNoise[i][j] for j in range(len(self.Signals[i]))]

    # Add an error readout here
    domain_list = ["time"]
    ch_list = ["ant1ch0", "ant1ch1", "ant2ch0", "ant2ch1", "ant3ch0", "ant3ch1"]
    for idom in range(len(domain_list)):
      print("\n \n Domain: ", domain_list[idom])
      for ich in range(len(ch_list)):
        print("\n Channel:", ch_list[ich])
        
        # Organization of nested lists:
        # Signals --> domains (time and freq) --> channels --> antennas --> traces

       # ## Uncomment if saving frequency data
       # # print("Number of events", len(self.Signals)) # should be 1
       #  print("Number of domains", len(self.Signals)) # should be 1
       #  print("Number of channels",  len(self.Signals[idom])) # should be 6
       
       #  print("Signals number of antennas", len(self.Signals[idom][ich])) # should be 163 (or 162 for ant3)
       #  print("NoiseOnly number of antennas", len(self.NoiseOnly[idom][ich])) # should be 163 (or 162 for ant3)
       #  print("SigPlusNoise number of antennas", len(self.SigPlusNoise[idom][ich])) # should be 163 (or 162 for ant3)

       #  print("Signals channel length", len(self.Signals[idom][ich][0])) # should be 4096
       #  print("NoiseOnly channel length", len(self.NoiseOnly[idom][ich][0])) # should be 4096
       #  print("SigPlusNoise channel length", len(self.SigPlusNoise[idom][ich][0])) # should be 4096

       ## Uncomment if only saving time domain data       
       # print("Number of events", len(self.Signals)) # should be 1
        print("Number of channels", len(self.Signals)) # should be 6
       
        print("Signals number of antennas", len(self.Signals[ich])) # should be 163 (or 162 for ant3)
        print("NoiseOnly number of antennas", len(self.NoiseOnly[ich])) # should be 163 (or 162 for ant3)
        print("SigPlusNoise number of antennas", len(self.SigPlusNoise[ich])) # should be 163 (or 162 for ant3)

        print("Signals channel length", len(self.Signals[ich][0])) # should be 1024
        print("NoiseOnly channel length", len(self.NoiseOnly[ich][0])) # should be 1024
        print("SigPlusNoise channel length", len(self.SigPlusNoise[ich][0])) # should be 1024

      # for ich2 in range(len(ch_list)-1):
      #   for iant in range((len(self.Signals[idom][ich2]))-1):
      #     if len(self.Signals[idom][ich2][iant]) != len(self.Signals[idom][ich2+1][iant]):
      #       log_warn("Channels {0} and {1} have different numbers of pure signal traces in the {2} domain".format(ch_list[ich2],ch_list[ich2+1],domain_list[idom]))
      #       log_warn("Num traces in {0}: {1}".format(ch_list[ich2],len(self.Signals[idom][ich2][iant])))
      #       log_warn("Num traces in {0}: {1}".format(ch_list[ich2+1],len(self.Signals[idom][ich2+1][iant])))
      #     if len(self.NoiseOnly[idom][ich2][iant]) != len(self.NoiseOnly[idom][ich2+1][iant]):
      #       log_warn("Channels {0} and {1} have different numbers of noise only traces in the {2} domain".format(ch_list[ich2],ch_list[ich2+1],domain_list[idom]))
      #       log_warn("Num traces in {0}: {1}".format(ch_list[ich2],len(self.NoiseOnly[idom][ich2][iant])))
      #       log_warn("Num traces in {0}: {1}".format(ch_list[ich2+1],len(self.NoiseOnly[idom][ich2+1][iant])))
      #     if len(self.SigPlusNoise[idom][ich2][iant]) != len(self.SigPlusNoise[idom][ich2+1][iant]):
      #       log_warn("Channels {0} and {1} have different numbers of noisy signal traces in the {2} domain".format(ch_list[ich2],ch_list[ich2+1],domain_list[idom]))
      #       log_warn("Num traces in {0}: {1}".format(ch_list[ich2],len(self.SigPlusNoise[idom][ich2][iant])))
      #       log_warn("Num traces in {0}: {1}".format(ch_list[ich2+1],len(self.SigPlusNoise[idom][ich2+1][iant])))
 
    # self.signalcounter = 0

    # for ich in range(len(ch_list)):
    #   SNR = self.SNRNoiseOnly[ich]
    #   print(f'SNR={SNR}, SNR type={type(SNR)}')
    #   if ich==0: #ant1ch0
    #     one_range = range(int(10*.5), int(10**1.35))    # These ranges were chosen by looking at the SNR distributions for each channel
    #     two_range = range(int(10**1.35), int(10**2.2))
    #     three_min = 10**2.2
    #   if ich==1: #ant1ch1
    #     one_range = range(int(10*.5), int(10**1.3))
    #     two_range = range(int(10**1.3), int(10**2.0))
    #     three_min = 10**2.0
    #   if ich==2: #ant2ch0
    #     one_range = range(int(10*.5), int(10**1.4))
    #     two_range = range(int(10**1.3), int(10**1.9))
    #     three_min = 10**1.9
    #   if ich==3: #ant2ch1
    #     one_range = range(int(10*.5), int(10**1.4))
    #     two_range = range(int(10**1.3), int(10**2.0))
    #     three_min = 10**2.0
    #   if ich==4: #ant3ch0
    #     one_range = range(int(10*.5), int(10**2.1))
    #     two_range = range(-1000,-999)
    #     three_min = 10**2.1
    #   if ich==5: #ant3ch1
    #     one_range = range(int(10*.5), int(10**1.7))
    #     two_range = range(int(10**1.7), int(10**2.7))
    #     three_min = 10**2.7

    #   ch_idx = self.snrsignalcounter % 6
    #   if SNR in two_range:
    #     self.bump2Traces[ch_idx] = np.append(bump2Traces, self.NoiseOnly[ich])
    #     self.bump2TraceROI[ch_idx] = np.append(bump2TraceROI, self.ROIList[ich])
    #     self.bump2TraceLength[ch_idx] = np.append(bump2TraceLength, self.TraceLengthList[ich])
    #   if SNR > three_min:
    #     self.bump3Traces[ch_idx] = np.append(bump3Traces, self.NoiseOnly[ich])
    #     self.bump3TraceROI[ch_idx] = np.append(bump3TraceROI, self.ROIList[ich])
    #     self.bump3TraceLength[ch_idx] = np.append(bump3TraceLength, self.TraceLengthList[ich])
      
    #   self.signalcounter += 1

  def Finish(self):
    self.Signals = np.array(self.Signals, dtype=object)
    self.SigPlusNoise = np.array(self.SigPlusNoise, dtype=object)
    self.NoiseOnly = np.array(self.NoiseOnly, dtype=object)
    # self.SNRSignals = np.array(self.SNRSignals)
    # self.SNRSigPlusNoise = np.array(self.SNRSigPlusNoise)
    # self.SNRNoiseOnly = np.array(self.SNRNoiseOnly)
    # self.bump2Traces = np.array(self.bump2Traces)
    # self.bump2TraceROI = np.array(self.bump2TraceROI)
    # self.bump2TraceLength = np.array(self.bump2TraceLength)
    # self.bump3Traces = np.array(self.bump3Traces)
    # self.bump3TraceROI = np.array(self.bump3TraceROI)
    # self.bump3TraceLength = np.array(self.bump3TraceLength)

    # ch_list = ["ant1ch0", "ant1ch1", "ant2ch0", "ant2ch1", "ant3ch0", "ant3ch1"]

    # for ich in range(len(ch_list)):
    #   # SNR
    #   np.save(OutputDir + "{0}_SNRSignals_{1}.npy".format(args.output, ch_list[ich]), self.SNRSignals[0][ich])
    #   np.save(OutputDir + "{0}_SNRSigPlusNoise_{1}.npy".format(args.output, ch_list[ich]), self.SNRSigPlusNoise[0][ich])
    #   np.save(OutputDir + "{0}_SNRNoiseOnly_{1}.npy".format(args.output, ch_list[ich]), self.SNRNoiseOnly[0][ich])

    #   # Bump 2
    #   np.save(OutputDir + Bump2/ + "{0}_Bump2Traces_{1}.npy".format(args.output, ch_list[ich]), self.bump2Traces[0][ich])
    #   np.save(OutputDir + Bump2/ + "{0}_Bump2TraceROI_{1}.npy".format(args.output, ch_list[ich]), self.bump2TraceROI[0][ich])
    #   np.save(OutputDir + Bump2/ + "{0}_Bump2TraceLength_{1}.npy".format(args.output, ch_list[ich]), self.bump2TraceLength[0][ich])

    #   # Bump 3
    #   np.save(OutputDir + Bump3/ + "{0}_Bump3Traces_{1}.npy".format(args.output, ch_list[ich]), self.bump3Traces[0][ich])
    #   np.save(OutputDir + Bump3/ + "{0}_Bump3TraceROI_{1}.npy".format(args.output, ch_list[ich]), self.bump3TraceROI[0][ich])
    #   np.save(OutputDir + Bump3/ + "{0}_Bump3TraceLength_{1}.npy".format(args.output, ch_list[ich]), self.bump3TraceLength[0][ich])


    ##########################################################################################################################
    # This should consolidate channel data for all events
    # ch_dict_signals = {"ant1ch0" : [[],[]], "ant1ch1" : [[],[]], "ant2ch0" : [[],[]], "ant2ch1" : [[],[]], "ant3ch0" : [[],[]], "ant3ch1" : [[],[]]}
    # ch_dict_noiseonly = {"ant1ch0" : [[],[]], "ant1ch1" : [[],[]], "ant2ch0" : [[],[]], "ant2ch1" : [[],[]], "ant3ch0" : [[],[]], "ant3ch1" : [[],[]]}
    # ch_dict_sigplusnoise = {"ant1ch0" : [[],[]], "ant1ch1" : [[],[]], "ant2ch0" : [[],[]], "ant2ch1" : [[],[]], "ant3ch0" : [[],[]], "ant3ch1" : [[],[]]}

    # print("\n Consolidating events for each channel:")
    # for ich, key in enumerate(ch_dict_signals):
    #   for ievent in range(len(self.Signals)):
    #     ch_dict_signals[key][0].append(self.Signals[ievent][0][ich])
    #     print(key)
    #     print("Number of air shower simulations read in: ", len(ch_dict_signals[key][0]))
    #     print("Number of antennas: ", len(ch_dict_signals[key][0][0]))
    #     print("Trace length: ", len(ch_dict_signals[key][0][0][0]))

    #     ch_dict_signals[key][1].append(self.Signals[ievent][1][ich])

    #     ch_dict_noiseonly[key][0].append(self.NoiseOnly[ievent][0][ich])
    #     ch_dict_noiseonly[key][1].append(self.NoiseOnly[ievent][1][ich])

    #     ch_dict_sigplusnoise[key][0].append(self.SigPlusNoise[ievent][0][ich])
    #     ch_dict_sigplusnoise[key][1].append(self.SigPlusNoise[ievent][1][ich])
    
    # ch_t_signals = [ch_dict_signals[key][0][0] for key in ch_dict_signals]
    # ch_t_noiseonly = [ch_dict_noiseonly[key][0][0] for key in ch_dict_noiseonly]
    # ch_t_sigplusnoise = [ch_dict_sigplusnoise[key][0][0] for key in ch_dict_sigplusnoise]

    # ch_f_signals = [ch_dict_signals[key][1][0] for key in ch_dict_signals]
    # ch_f_noiseonly = [ch_dict_noiseonly[key][1][0] for key in ch_dict_noiseonly]
    # ch_f_sigplusnoise = [ch_dict_sigplusnoise[key][1][0] for key in ch_dict_sigplusnoise]

    ##########################################################################################################################
    # Counting number of pure signal and noise only traces
    SigTraces = [0,0,0,0,0,0]
    NoiseTraces = [0,0,0,0,0,0]

    for i in range(len(self.Signals)):
      channel = self.Signals[i]
      for trace in channel:
        if np.max(trace) == 0:
          NoiseTraces[i] += 1
        else:
          SigTraces[i] += 1

    print("\nTotal Number of Signal Traces (for this run): ", sum(SigTraces))
    print("Total Number of Noise Only Traces (for this run): ", sum(NoiseTraces))

    # print(f"Signals trace example: {self.Signals[0][0]}")
    # print(f"NoiseOnly trace example: {self.NoiseOnly[0][0]}")
    # print(f"SigPlusNoise trace example: {self.SigPlusNoise[0][0]}")

    ##########################################################################################################################

    # Paula's
    # OutputDir = f"/data/user/pgalvezm/TrainingAndTesting/data-production/taxi-noise/data/Dataset_{self.dataset}/"
    # Dana's 
    OutputDir = f"/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_{self.dataset}/"
    # OutputDir = ABS_PATH_HERE + f"/../data/Dataset_{self.dataset}/"
    ch_list = ["ant1ch0", "ant1ch1", "ant2ch0", "ant2ch1", "ant3ch0", "ant3ch1"]

    for ich in range(len(ch_list)):

      # RMS (noise only)
      # np.save(OutputDir + "{0}_RMS.npy".format(self.arguments, ch_list[ich]), self.RMSList)

      # time series
      # np.save(OutputDir + "{0}_Time_{1}_Signals.npy".format(self.arguments, ch_list[ich]), self.Signals[ich])
      # np.save(OutputDir + "{0}_Time_{1}_SigPlusNoise.npy".format(self.arguments, ch_list[ich]), self.SigPlusNoise[ich])
      # np.save(OutputDir + "{0}_Time_{1}_NoiseOnly.npy".format(self.arguments, ch_list[ich]), self.NoiseOnly[ich])
      np.save(OutputDir + "{0}_{1}_Signals.npy".format(self.arguments, ch_list[ich]), self.Signals[ich])
      np.save(OutputDir + "{0}_{1}_SigPlusNoise.npy".format(self.arguments, ch_list[ich]), self.SigPlusNoise[ich])
      np.save(OutputDir + "{0}_{1}_NoiseOnly.npy".format(self.arguments, ch_list[ich]), self.NoiseOnly[ich])

      # frequency series

      # np.save(OutputDir + "{0}_Spectrum_{1}_Signals.npy".format(self.arguments, ch_list[ich]), self.Signals[1][ich])
      # np.save(OutputDir + "{0}_Spectrum_{1}_SigPlusNoise.npy".format(self.arguments, ch_list[ich]), self.SigPlusNoise[1][ich])
      # np.save(OutputDir + "{0}_Spectrum_{1}_NoiseOnly.npy".format(self.arguments, ch_list[ich]), self.NoiseOnly[1][ich])

      # np.save(OutputDir + "{0}_Freqs_{1}_Signals.npy".format(self.arguments, ch_list[ich]), self.Signals[2][ich])
      # np.save(OutputDir + "{0}_Freqs_{1}_SigPlusNoise.npy".format(self.arguments, ch_list[ich]), self.SigPlusNoise[2][ich])
      # np.save(OutputDir + "{0}_Freqs_{1}_NoiseOnly.npy".format(self.arguments, ch_list[ich]), self.NoiseOnly[2][ich])

    print("Finishing up...")
