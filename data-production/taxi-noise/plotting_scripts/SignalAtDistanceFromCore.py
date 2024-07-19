#!/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

from I3Tray import *
from icecube import icetray, radcube, dataio, dataclasses
from icecube.dataclasses import I3AntennaGeo
from icecube.icetray import I3Units
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, nargs='+', default=[], help='Input data files.')
args = parser.parse_args()

class RadcubePlotter(icetray.I3Module):
  def __init__(self,ctx):
    icetray.I3Module.__init__(self,ctx)
    self.fig = 0
    self.gs = 0
    self.gsCount = 0;
    self.rPos = []
    self.outputName = "OutputPlot.pdf"

  def I3RadVector3DToPython(self, vectorMap):
    Times, Ex, Ey, Ez, Freqs, Fx, Fy, Fz =  [], [], [], [], [], [], [], []
    for antkey in vectorMap.keys():
        vec3d = vectorMap[antkey]
        fftData = dataclasses.FFTData3D() ## make fftData container

        fftData.LoadTimeSeries(vec3d)
        timeseries = fftData.GetTimeSeries()
        spectrum = fftData.GetFrequencySpectrum()
        times, E_x, E_y, E_z = radcube.RadTraceToPythonList(timeseries)
        freqs, specX, specY, specZ = radcube.RadTraceToPythonList(spectrum)
        
        Times.append(times / I3Units.nanosecond)
        Ex.append(E_x)
        Ey.append(E_y)
        Ez.append(E_z)
        Freqs.append(freqs / I3Units.megahertz)
        Fx.append(specX)
        Fy.append(specY)
        Fz.append(specZ)

    return Times, Ex, Ey, Ez, Freqs, Fx, Fy, Fz

  def DAQ(self,frame):
    ## Extracting data from frame
    if frame.Has("CoREASEFieldMap"):
      rawDataMap = frame["CoREASEFieldMap"] #Get the simulated traces
    else:
      log_fatal("No fields found in the frame")

    if frame.Has("CoREASPrimary"):
      primary = frame["CoREASPrimary"] #Get the primary particle direction
    else:
      log_fatal("No primary found in the frame")
    showerDir = primary.dir
    
    if frame.Has("I3AntennaGeometry"):
      geomap = frame["I3AntennaGeometry"].antennageo  #Get the antenna Geometry
    else:
      log_fatal("No geometry found in the frame")


    #Get the antenna locations and save them
    for antkey in rawDataMap.keys():   
      antennaLoc = geomap[antkey].position
      antennaLoc.z -= primary.pos.z #The IC origin is in the middle of in-ice, move it first to surface
      antennaLoc = radcube.GetMagneticFromIC(antennaLoc, showerDir)
      xPos = antennaLoc.x / I3Units.m
      yPos = antennaLoc.y / I3Units.m
      self.rPos.append(np.sqrt(xPos**2 + yPos**2))

    print("-----------------")
    print("-----------------")
    print("-----------------")
    print("printing stuff")
    print("showerDir: ", showerDir)
    print("-----------------")
    print("-----------------")
    print("-----------------")

    FilteredMap = frame["FilteredEField"]
    print("R positions of antennas:")
    for i in range(100):
      print("R = ", self.rPos[i], "Antind =", i)
    # Times, Ex, Ey, Ez, Freqs, Fx, Fy, Fz = self.I3RadVector3DToPython(rawDataMap)
    Times, Ex, Ey, Ez, Freqs, Fx, Fy, Fz = self.I3RadVector3DToPython(FilteredMap)

    # Making Plots
    NRows = 3
    NCols = 2
    gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
    fig = plt.figure(figsize=(6*NCols, 5*NRows))

    ax = fig.add_subplot(gs[0])
    ax.set_title("R = 1m")
    ant = 96
    ax.plot(Ex[ant], label='Ex')
    ax.plot(Ey[ant], label='Ey')
    ax.plot(Ez[ant], label='Ez')
    ax.legend(loc='best', prop={'size': 8})
    ax.set_xlabel("Time")
    ax.set_xlim(2200, 2800)

    ax = fig.add_subplot(gs[1])
    ax.set_title("R = 1m")
    ax.plot(Freqs[ant], np.abs(Fx[ant]), label='Fx')
    ax.plot(Freqs[ant], np.abs(Fy[ant]), label='Fy')
    ax.plot(Freqs[ant], np.abs(Fz[ant]), label='Fz')
    ax.legend(loc='best', prop={'size': 8})
    ax.set_xlabel("Freq [MHz]")

    ax = fig.add_subplot(gs[2])
    ax.set_title("R = 145m")
    ant = 15
    ax.plot(Ex[ant], label='Ex')
    ax.plot(Ey[ant], label='Ey')
    ax.plot(Ez[ant], label='Ez')
    ax.legend(loc='best', prop={'size': 8})
    ax.set_xlabel("Time")
    ax.set_xlim(2200, 2800)

    ax = fig.add_subplot(gs[3])
    ax.set_title("R = 145m")
    ax.plot(Freqs[ant], np.abs(Fx[ant]), label='Fx')
    ax.plot(Freqs[ant], np.abs(Fy[ant]), label='Fy')
    ax.plot(Freqs[ant], np.abs(Fz[ant]), label='Fz')
    ax.legend(loc='best', prop={'size': 8})
    ax.set_xlabel("Freq [MHz]")

    ax = fig.add_subplot(gs[4])
    ax.set_title("R = 500m")
    ant = 87
    ax.plot(Ex[ant], label='Ex')
    ax.plot(Ey[ant], label='Ey')
    ax.plot(Ez[ant], label='Ez')
    ax.legend(loc='best', prop={'size': 8})
    ax.set_xlabel("Time")
    ax.set_xlim(2000, 3500)

    ax = fig.add_subplot(gs[5])
    ax.set_title("R = 500m")
    ax.plot(Freqs[ant], np.abs(Fx[ant]), label='Fx')
    ax.plot(Freqs[ant], np.abs(Fy[ant]), label='Fy')
    ax.plot(Freqs[ant], np.abs(Fz[ant]), label='Fz')
    ax.legend(loc='best', prop={'size': 8})
    ax.set_xlabel("Freq [MHz]")
    OutputDir = "/home/dkullgren/work"
    fig.savefig(OutputDir + "/TestPlot.pdf", bbox_inches='tight')


icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
tray = I3Tray()
################################
## Add modules
################################
tray.AddModule("CoreasReader", "coreasReader",
               DirectoryList=args.input,
               MakeGCDFrames=True,
               MakeDAQFrames=True,
              )
tray.AddModule("ZeroPadder", "iPad",
               InputName=radcube.GetDefaultSimEFieldName(),
               OutputName="ZeroPaddedMap",
               ApplyInDAQ = True,
               AddToFront = True,
               AddToTimeSeries = True,
               FixedLength = int(4096 * (5/4))
              )

tray.AddModule("TraceResampler", "Resampler",
               InputName="ZeroPaddedMap",
               OutputName="ResampledMap",
               ResampledBinning=0.25*I3Units.ns
              )
tray.AddModule("BandpassFilter","BoxFilterToEfield",
               InputName="ResampledMap", 
               OutputName="FilteredEField",
               FilterType=radcube.eBox,
               FilterLimits=[200* I3Units.megahertz,1200* I3Units.megahertz],
               ApplyInDAQ = True,
               )
tray.AddModule(RadcubePlotter, "thePlotter")

tray.Execute()