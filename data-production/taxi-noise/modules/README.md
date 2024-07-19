This folder contains modules used in MakeTraces.py. <br>

BackgroundTimeFilter.py - Only chooses background waveforms from between given times to avoid overly noisy recording periods <br>
DataExtractor.py - Extracts and saves all data <br>
OneBinCleaner.py - Cleans up some of the artifacts in the noise data (no longer used) <br>
PreSNRCutSignalPlotter.py - Plots signals before SNR cut is applied to choose best SNR cutoff <br>
PrintArray.py - Prints given array to check how waveforms change through the modules <br>
SelectCleanSig.py - Rejects all simulated signals under a given SNR cutoff so only the "pretty" signals are kept <br>
ShiftAntTraces.py - Shifts signals randomly within a certain range
