from I3Tray import I3Tray
from icecube import icetray, radcube, dataclasses
from icecube.icetray.i3logging import log_info, log_error, log_fatal, log_trace
icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
import numpy as np

class OneBinCleaner(icetray.I3Module):

  def __init__(self,ctx):
    icetray.I3Module.__init__(self,ctx)
    self.inputNameNoisy = ""
    self.AddParameter("InputNameNoisy", "Waveform with background", self.inputNameNoisy)
    self.inputNameNoise = ""
    self.AddParameter("InputNameNoise", "TAXI background", self.inputNameNoise)
    self.outputNameNoisy = ""
    self.AddParameter("OutputNameNoisy", "Cleaned waveform with background", self.outputNameNoisy)
    self.outputNameNoise = ""
    self.AddParameter("OutputNameNoise", "Cleaned TAXI background", self.outputNameNoise)
    self.applyInDAQ = True
    self.AddParameter("ApplyInDAQ", "If true, will apply filter on Q Frames, else P Frames", self.applyInDAQ)

  def Configure(self):
    log_info("Configuring " + self.name)
    self.inputNameNoisy = self.GetParameter("InputNameNoisy")
    self.inputNameNoise = self.GetParameter("InputNameNoise")
    self.outputNameNoisy = self.GetParameter("OutputNameNoisy")
    self.outputNameNoise = self.GetParameter("OutputNameNoise")
    self.applyInDAQ = self.GetParameter("ApplyInDAQ")
    log_info("Will apply in the "+("P", "Q")[self.applyInDAQ]+ " Frame")

  def RunOnFrame(self, frame):
    log_trace("({0}) Cleaning Zero Bins".format(self.name))
    if not self.inputNameNoisy in frame.keys():
      log_warn("Did not find AntennaDataMap named {0} in the frame".format(self.inputName))
      return
    antDataMapNoisy = frame[self.inputNameNoisy]
    antDataMapNoise = frame[self.inputNameNoise]
    outputAntDataMapNoisy = dataclasses.I3AntennaDataMap()
    outputAntDataMapNoise = dataclasses.I3AntennaDataMap()
    for iant, antkey in enumerate(antDataMapNoisy.keys()):
      chMapNoisy = antDataMapNoisy[antkey]
      chMapNoise = antDataMapNoise[antkey]
      outputChMapNoisy = dataclasses.I3AntennaChannelMap()
      outputChMapNoise = dataclasses.I3AntennaChannelMap()
      for ich, chkey in enumerate(chMapNoisy.keys()):
        chDataNoisy = chMapNoisy[chkey]
        chDataNoise = chMapNoise[chkey]
        fftNoisy = chDataNoisy.GetFFTData()
        fftNoise = chDataNoise.GetFFTData()
        noisySeries = fftNoisy.GetTimeSeries()
        noiseSeries = fftNoise.GetTimeSeries()
        if len(noisySeries) != len(noiseSeries):
          print("Warning!", len(noisySeries), len(noiseSeries))
        assert(len(noisySeries) == len(noiseSeries))
        if noisySeries.binning != noiseSeries.binning:
          print("Warning!", noisySeries.binning, noiseSeries.binning)
        assert(noisySeries.binning == noiseSeries.binning)

        std = radcube.GetStd(noiseSeries, 1.)
        mean = radcube.GetMean(noiseSeries, 1.)
        cutoff_sigma = 6
        
        bad_index = []
        for ibin in range(len(noiseSeries)):
          if abs(((noiseSeries[ibin] - mean) / std)) > cutoff_sigma:
            bad_index.append(ibin)

        for ibin in range(len(noiseSeries)):
          for ibin in bad_index:
            j = (ibin + 1) % len(noiseSeries)
            while j in bad_index:
              j = (j + 1) % len(noiseSeries)
            h = (ibin - 1) % len(noiseSeries)
            while h in bad_index:
              h = (h - 1) % len(noiseSeries)
            if ibin > 1022:                     # final bin
              x_avg = noiseSeries[h]
            elif ibin < 1:                      # first bin
              x_avg = noiseSeries[j]
            else:
              x_avg = (noiseSeries[h] + noiseSeries[j]) / 2.
            voltage_shift = x_avg - noiseSeries[ibin]
            noiseSeries[ibin] = x_avg
            noisySeries[ibin] += voltage_shift

          outputFFTNoisy = dataclasses.FFTData(noisySeries)
          outputFFTNoise = dataclasses.FFTData(noiseSeries)
        outputChMapNoisy[chkey] = dataclasses.I3AntennaChannel(outputFFTNoisy)
        outputChMapNoise[chkey] = dataclasses.I3AntennaChannel(outputFFTNoise)
      outputAntDataMapNoisy[antkey] = outputChMapNoisy
      outputAntDataMapNoise[antkey] = outputChMapNoise
    frame[self.outputNameNoisy] = outputAntDataMapNoisy
    frame[self.outputNameNoise] = outputAntDataMapNoise

  def DAQ(self, frame):
    if self.applyInDAQ:
      self.RunOnFrame(frame)
    self.PushFrame(frame)

  def Physics(self, frame):
    if not self.applyInDAQ:
      self.RunOnFrame(frame)
    self.PushFrame(frame)