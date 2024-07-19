#!/usr/bin/env python3

from icecube.icetray import I3Tray, I3Units
# from I3Tray import I3Tray      ## this is depreciated
from icecube import icetray, radcube, dataclasses, taxi_reader
from icecube.radcube import defaults
# from icecube.icetray import I3Units
import random
from icecube.icetray.i3logging import log_info, log_error, log_fatal, log_trace, log_warn
icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
from modules.SNRCut import SNRCut
from modules.ShiftTraces import ShiftTraces
from modules.PrintArray import PrintArray
from modules.SelectCleanSig import SelectCleanSig
from modules.ShiftAntTraces import ShiftAntTraces
from modules.OneBinCleaner import OneBinCleaner
from modules.DataExtractor import DataExtractor
from modules.SignalCombiner import SignalCombiner
from modules.SNRSeparator import SNRSeparator
# from modules.PreSNRCutSignalPlotter import PreSNRCutSignalPlotter
import datetime as datetime
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, nargs='+', default=[], help='Input data files.')
parser.add_argument('--output', type=str, default="Run", help='Output i3 file name.')
parser.add_argument('--taxi', type=str, nargs='+',  default=[], help='Location of taxi backround files')

args = parser.parse_args()
assert(len(args.input))

print(f'\nargs = {args}')
print(f'args.output = {args.output}\n')

## 11/23, 01/24: current simulation sampling rate is 0.2 ns (found in sim files) and background noise is currently 1.25 ns

TAXI_sampling = 800 # MHz
# TAXI_sampling = 1000 # MHz

NBins = 1000    #Number of bins at 1ns sampling
# NBins = 1250    #Number of bins at 1ns sampling
if TAXI_sampling==800: 
  resampled_binning=1.25
  fixed_length = 6250
elif TAXI_sampling==1000:
  resampled_binning=1
  fixed_length = int(5*NBins)

upsampleFactor = 2 # This gives a final sampling rate of 0.5ns
filterLimits = [110* I3Units.megahertz, 185* I3Units.megahertz]
# hourRange = [datetime.time(0,0), datetime.time(12,0)]     ## Define the range of hours per day to be used
snr_cutoff = 0
# snr_cutoff = 200

dataset='Auger_v4'
new_elec_resp = True
NeedToUpdateSNRCutoff = False ## Listing this as True will plot signals before the SNR cutoff so a new SNR cutoff can be determined for the new dataset


# add electronics
tray = I3Tray()

AntennaServiceName = defaults.CreateDefaultAntennaResponse(tray)

if new_elec_resp:
  electronicName = "electronicName"

  tray.AddService(
      "I3ElectronicsResponseFactory",
      electronicName,
      IncludeLNA=True,
      IncludeCables=True,    ## ON
      CableTemperature=radcube.constants.cableTemp,
      IncludeRadioBoard=False, ## OFF
      IncludeTaxi=False,  ## OFF
      IncludeDAQ=True,
      InstallServiceAs=electronicName
  )
else:
  electronicName = defaults.CreateDefaultElectronicsResponse(tray,"ElectronicServiceName")

tray.AddModule("CoreasReader", "coreasReader", # sampling rate  = .2 ns
               DirectoryList=args.input,
               MakeGCDFrames=True,
               MakeDAQFrames=True
              )

# "GeneratedNoiseMap" # OutPut name of BringTheNoise Module

tray.AddModule("ZeroPadder", "iPad", # Add zeroes to make traces same length (in time domain)
               InputName=radcube.GetDefaultSimEFieldName(),
               OutputName="ZeroPaddedMap",
               ApplyInDAQ = True,
               AddToFront = True,
               AddToTimeSeries = True,
              #  FixedLength = int(5 * NBins)
               FixedLength = fixed_length
              )

tray.AddModule("ChannelInjector", "ChannelInjector", # Inject antenna response
                InputName="ZeroPaddedMap",
                OutputName="ChannelInjectedMap",
                AntennaResponseName=AntennaServiceName
              )

tray.AddModule("TraceResampler", "Resampler", # .2ns -> 1ns or 1.25ns
               InputName="ChannelInjectedMap",
               OutputName="ResampledVoltageMap",
               ResampledBinning=resampled_binning*I3Units.ns
              )

tray.AddModule("AddPhaseDelay", "AddPhaseDelay", # moves pulse to middle
                InputName="ResampledVoltageMap",
                OutputName="PhaseDelayed",
                ApplyInDAQ=True,
                DelayTime=-120*I3Units.ns
              )

# tray.AddModule(PrintArray, "PrintArray",
#                InputName="PhaseDelayed",
#                ApplyInDAQ=True
#               )

# if NeedToUpdateSNRCutoff:  ## Plot signals here to find SNR cutoff
#   tray.AddModule(PreSNRCutSignalPlotter, "PreSNRCutSignalPlotter",
#                 InputName="PhaseDelayed",
#                 SNRCutoffValue=snr_cutoff,
#                 PlotAroundCutoff = True,  ## True if want to plot signals with SNRs near cutoff, False if want to plot random waveforms
#                 Dataset=dataset
#                 )

tray.AddModule(SelectCleanSig, "SelectOnlyCleanSig",
                InputName="PhaseDelayed",
                OutputName="CleanSignals",
                SNRCutoffValue=snr_cutoff,    # this value will be updated after plotting the SNRs of the PhaseDelayed traces
                ApplyInDAQ=True,
              )

tray.AddModule(PrintArray, "PrintArray",
               InputName="CleanSignals",
               ApplyInDAQ=True
              )

tray.AddModule(ShiftAntTraces, "ShiftAntTraces",
               InputName="CleanSignals",
               OutputName="ShiftedSignals",
               ApplyInDAQ=True,
               )


tray.AddModule("BandpassFilter", "BandpassFilterBox",
                 InputName="ShiftedSignals",
                 OutputName="FilteredConvolvedSignal",
                 FilterType=radcube.eBox,
                 FilterLimits=filterLimits,
                 # ButterworthOrder=13,
                 ApplyInDAQ=True
                )

tray.AddModule("ElectronicResponseAdder", "AddElectronics", # LNAs, etc.
               InputName="ShiftedSignals",
               OutputName="ConvolvedSignal",
               ElectronicsResponse=electronicName
              )
#   )

tray.AddModule("WaveformDigitizer", "waveformdigitizer",
               InputName="ConvolvedSignal",
               # InputName="ShiftedSignal",
               OutputName="DigitizedSignal",
               ElectronicsResponse=electronicName
              )

## Add module here to make sure TAXI files are from the correct times and then read them into MeasuredNoiseAdder
# tray.AddModule(BackgroundTimeFilter, "BackgroundTimeFilter",
#               TaxiFile=args.taxi,
#               OutputName="timeFilteredBackground",
#               HourRange=hourRange
#               )

tray.AddModule(radcube.modules.MeasuredNoiseAdder, "AddTaxiBackgroundTrace",   ## Need to change the variable hourRange in TAXIBackgroundReader if you want to get data from a new range of hours
               InputName="DigitizedSignal",
               OutputName="WaveformWithBackground",
               TaxiFile=args.taxi,
               NTimeBins=NBins,
               NTraces=60000,
               ConvertToVoltage = False,
               InsertNoiseOnly=True,
               Overuse=False,

               RequiredWaveformLength=1024,   # may need to update radcube to use this
                
               # Arguments below have been added to clean the traces
               RemoveBinSpikes = True,
               BinSpikeDeviance = 800,
               RemoveNegativeBins = True,
               MedianOverCascades = True
              )

# tray.AddModule(OneBinCleaner, "OneBinCleaner",
#                InputNameNoisy="WaveformWithBackground",
#                InputNameNoise="TAXINoiseMap",
#                OutputNameNoisy="CleanedNoisy",
#                OutputNameNoise="CleanedNoise"
#                )

#for framename in ['TrueSignal', 'WaveformWithBackground', 'TAXINoise']:


tray.AddModule("I3NullSplitter","splitter", # moves from DAQ to DAQ and physics
               SubEventStreamName="RadioEvent"
                ) 


for framename in ['WaveformWithBackground', 'TAXINoiseMap']: # remove antenna response after combining with background
  tray.AddModule("PedestalRemover", "pedestalRemover{0}".format(framename),
               InputName="{0}".format(framename),
               OutputName="{0}Voltage".format(framename),
               ElectronicsResponse=electronicName,          #Name of I3ElectronicsResponse service
               ConvertToVoltage=True
              )

for framename in ['WaveformWithBackgroundVoltage', 'TAXINoiseMapVoltage']:
  tray.AddModule("BandpassFilter", "BandpassFilterBoxForNoisy{0}".format(framename),
                 InputName="{0}".format(framename),
                 OutputName="Filtered{0}".format(framename),
                 FilterType=radcube.eBox,
                 FilterLimits=filterLimits,
                 # ButterworthOrder=13,
                 ApplyInDAQ=False
                )

for framename in ['FilteredWaveformWithBackgroundVoltage', 'FilteredTAXINoiseMapVoltage']:
  tray.AddModule("ElectronicResponseRemover", f"RemoveElectronics{framename}",
               InputName=f"{framename}",
               OutputName=f"Deconvolved{framename}",
               ElectronicsResponse=electronicName
              )

# Uncomment to change upsampling rate from 1.25 ns
#for framename in ['DeconvolvedFilteredWaveformWithBackgroundVoltage', 'DeconvolvedFilteredTAXINoiseMapVoltage', 'DeconvolvedShiftedFilteredSignal']:
# for framename in ['FilteredCleanedNoisyVoltage', 'FilteredCleanedNoiseVoltage', 'ShiftedFilteredScaledSignal']:
#  tray.AddModule("ZeroPadder", "iPadSig{0}".format(framename), # 1 ns -> whatever we want (currently .25 ns)
#               InputName=framename,
#               OutputName="The{0}".format(framename),
#               ApplyInDAQ = False,
#               AddToFront = False,
#               AddToTimeSeries = False,
#               AppendN = int(NBins/2) * (upsampleFactor - 1)
#              )

tray.AddModule(DataExtractor, "DataExtract",
               InputNameNoisy="DeconvolvedFilteredWaveformWithBackgroundVoltage",
               InputNameNoise="DeconvolvedFilteredTAXINoiseMapVoltage",
               InputNameTrue="FilteredConvolvedSignal",
               Arguments=args.output,
               Dataset=dataset,
              )

tray.AddModule("TrashCan", "trashcan")
tray.Execute()
tray.Finish()
