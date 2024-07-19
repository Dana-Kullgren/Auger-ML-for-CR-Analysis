from I3Tray import I3Tray
from icecube import icetray, radcube, dataclasses
from icecube.icetray.i3logging import log_info, log_error, log_fatal, log_trace
icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
import numpy as np
import random

class PrintArray(icetray.I3Module):

  def __init__(self,ctx):
    icetray.I3Module.__init__(self,ctx)
    self.inputName = ""
    self.AddParameter("InputName", "Input Antenna DataMap", self.inputName)
    self.applyInDAQ = True
    self.AddParameter("ApplyInDAQ", "If true, will apply filter on Q Frames, else P Frames", self.applyInDAQ)

  def Configure(self):
    log_info("Configuring " + self.name)
    self.inputName = self.GetParameter("InputName")
    self.applyInDAQ = self.GetParameter("ApplyInDAQ")
    log_info("Will apply in the "+("P", "Q")[self.applyInDAQ]+ " Frame")


  def RunOnFrame(self, frame):
    log_trace("({0}) Printing the length of the given input".format(self.name))
    
    antDataMap = frame[self.inputName]
    log_info("({0}) len(antDataMap)={1}".format(self.name, len(antDataMap)))
    outputAntDataMap = dataclasses.I3AntennaDataMap()

    for iant, antkey in enumerate(antDataMap.keys()):
      chMap = antDataMap[antkey]
    #   log_info("({0}) antkey={1}".format(self.name, antkey))
    #   for ich, chkey in enumerate(chMap.keys()):
        # log_info("({0}) chkey={1}".format(self.name, chkey))
        
  def DAQ(self, frame):
    if self.applyInDAQ:
      self.RunOnFrame(frame)
    self.PushFrame(frame)

  def Physics(self, frame):
    if not self.applyInDAQ:
      self.RunOnFrame(frame)
    self.PushFrame(frame)
