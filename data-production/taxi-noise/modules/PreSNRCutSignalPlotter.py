from I3Tray import I3Tray
from icecube import icetray, radcube, dataclasses
from icecube.icetray.i3logging import log_info, log_error, log_fatal, log_trace
icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from icecube.icetray import I3Units

class PreSNRCutSignalPlotter(icetray.I3Module):

  def __init__(self,ctx):
    icetray.I3Module.__init__(self,ctx)
    self.inputName = ""
    self.AddParameter("InputName", "Input Antenna DataMap", self.inputName)
    self.dataset = ""
    self.AddParameter("Dataset", "Version of current dataset", self.dataset)
    self.snr_cutoff = ""
    self.AddParameter("SNRCutoffValue", "Will plot traces above and below cutoff value", self.snr_cutoff)
    self.plot_around_cutoff = ""
    self.AddParameter("PlotAroundCutoff", "If True, will plot traces above and below cutoff value", self.plot_around_cutoff)

  def Configure(self):
    log_info("Configuring " + self.name)
    self.inputName = self.GetParameter("InputName")
    self.dataset = self.GetParameter("Dataset")
    self.snr_cutoff = self.GetParameter("SNRCutoffValue")
    self.plot_around_cutoff = self.GetParameter("PlotAroundCutoff")
    self.Signals = [[] for i in range(3*2)]
    self.signalcounter = 0

  def DAQ(self, frame):
    log_trace("({0}) Plotting signals".format(self.name))

    ## The following function assumes that the signals are located at the center of traces
    def GetSNR(trace, binlow=510):
      Trace_Peak = np.max(np.abs(trace)) 
      chunk = np.array(trace[binlow:810])
      RMS_squared = sum(chunk ** 2) / len(chunk)
      SNR = Trace_Peak ** 2 / RMS_squared
      return SNR

    path = f"/home/dkullgren/work/WaveformML-DataOnly/data-production/taxi-noise/data/Dataset_{self.dataset}/"
    Signals = frame[self.inputName] # Its an antenna data map

    def GetAntData(AntMap, List, counter): ## give it a Antenna Map and empty list, it will give you list of time series
      nant = 3
      nch = 2 # Under the assumption that all antennas have 2 channels

      for iant, antkey in enumerate(AntMap.keys()):
        channelMapSig = AntMap[antkey]
        for ich, key in enumerate(channelMapSig.keys()):
          chdata = channelMapSig[key]
          fft = chdata.GetFFTData()
          timeseries = fft.GetTimeSeries()
          timeseriespy = [timeseries[i] for i in range(timeseries.GetSize())]
    
          List[counter % 6].append(timeseriespy)
          counter += 1
      return counter

    ch_list = ["ant1ch0", "ant1ch1", "ant2ch0", "ant2ch1", "ant3ch0", "ant3ch1"]
    self.signalcounter = GetAntData(Signals, self.Signals, self.signalcounter) #Should get 976 (nant*nch)

    # print(f'self.Signals = {self.Signals}')

    # Plotting starts here
    NRows = 2
    NCols = 3
    waveformTypes = 1 ## We are only plotting pure signals so this is 1
    plot_title = 'PreSNRCutSignal'   ## the title of the plot depends on the types of waveforms being plotted

    # plotindex = 3 # this is index of the trace I am plotting from the file (I chose 0 so I could plot once trace for each ant)
    # while np.max(np.abs(hilbert(s[0][plotindex]))) / I3Units.mV > cutoff:
    #   plotindex += 1

    if plot_around_cutoff:  ## True if want to plot signals with SNRs near cutoff
      plotindex = 0  ## this is index of the trace I am plotting from the file
      for i in range(5):

        gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
        fig = plt.figure(figsize=(6*NCols, 5*NRows))

        SNRNearCutoff = False
        while SNRNearCutoff == False
          plot_index += 1
          ## Calculate SNR values for this plot index
          ch_snrs = [GetSNR(self.Signals[ich][plotindex]) for ich in range(len(ch_list))]
          if (ch_snrs[ich] < snr_cutoff+50) and (ch_snrs[ich] > snr_cutoff-50):
            SNRNearCutoff = True

        for ich in range(len(ch_list)):
          ch = ch_list[ich]
          plot_position = 0 ## this will determine where the newest axis will be; increments after each axis
          sigPeak = np.max(np.abs(hilbert(self.Signals[ich][plotindex]))) / I3Units.mV # Can also use abs value instead of hilbert 

          ax = fig.add_subplot(gs[plot_position+ich*waveformTypes])
          plot_position += 1
          print(f'ax[ich][0] = ax[{ich}][{0}]')
          x1 = np.array(self.Signals[ich][plotindex]) / I3Units.mV
          print(f'ch={ch}, plotindex={plotindex}, x1[0]={x1[0]}')
          ax.plot(x1, label = "Pure Signal")
          ax.set_xlabel("Time [ns]", fontsize=16)
          ax.set_ylabel(r"Amplitude [mV]", fontsize=16)
          ax.legend(loc='best', prop={'size': 12})
          # ax.set_title(f'{ch}, SigPeak={sigPeak:.3f}', fontsize=18)
          ax.set_title(f'{ch}, SNR={GetSNR(self.Signals[ich][plotindex]):.3f}', fontsize=18)

          plot_index += 1

        fig.suptitle('Traces', fontsize=24)  
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(path + f"/TracesPlot_{plot_title}{self.dataset}_plotindex={plotindex}.pdf", bbox_inches='tight')

    else:  ## False if want to plot random waveforms
      for plotindex in range(5):  ## this is index of the trace I am plotting from the file
        gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
        fig = plt.figure(figsize=(6*NCols, 5*NRows))

        for ich in range(len(ch_list)):
          ch = ch_list[ich]
          plot_position = 0 ## this will determine where the newest axis will be; increments after each axis
          sigPeak = np.max(np.abs(hilbert(self.Signals[ich][plotindex]))) / I3Units.mV # Can also use abs value instead of hilbert 

          ax = fig.add_subplot(gs[plot_position+ich*waveformTypes])
          plot_position += 1
          print(f'ax[ich][0] = ax[{ich}][{0}]')
          x1 = np.array(self.Signals[ich][plotindex]) / I3Units.mV
          print(f'ch={ch}, plotindex={plotindex}, x1[0]={x1[0]}')
          ax.plot(x1, label = "Pure Signal")
          ax.set_xlabel("Time [ns]", fontsize=16)
          ax.set_ylabel(r"Amplitude [mV]", fontsize=16)
          ax.legend(loc='best', prop={'size': 12})
          # ax.set_title(f'{ch}, SigPeak={sigPeak:.3f}', fontsize=18)
          ax.set_title(f'{ch}, SNR={GetSNR(self.Signals[ich][plotindex]):.3f}', fontsize=18)

        fig.suptitle('Traces', fontsize=24)  
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(path + f"/TracesPlot_{plot_title}{self.dataset}_plotindex={plotindex}.pdf", bbox_inches='tight')
