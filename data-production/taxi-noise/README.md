BinToI3File.py - Convert TAXI binary files to I3 files. Run with following command (use no file extension for the OUTPUT_NAME): <br>
  `./BinToI3File_UsingTAXIScripts.py BIN_FILE_NAME --output OUTPUT_NAME` <br>
ChannelConsolidator.py - Takes channel-specific files from DatasetConsolidator and combines them <br>
ChannelSNRFileCreator.py - Calculates SNR for each channel AND for dataset overall <br>
ClassifierTracesSNRCalculator.py - Calculates SNR values for (noise and noisy) traces <br>
DatasetConsolidator.py - Creates a large dataset from the files produced by MakeTraces <br>
DatasetShapeFinder.py - Prints the shape of a files of traces <br>
DenoiserNoisySNRCalculator.py - Calculates SNR values for noisy traces <br>
MakeTracesAuger.py - Creates a dataset of pure signals, noise only traces (from Auger), and noisy traces (MakeTraces modules are found in the `modules` folder) <br>
MakeTracesAuger.sh, MakeTracesAuger.sub - Bash and submission scripts to submit MakeTraces jobs to condor <br>
NoiseOnlyTraceFinder.py - Takes data which is prepared for testing on the Classifier (ex: traces_train.py) and produces files for NoiseOnly and SigPlusNoise <br>
SigPeakCutoff.py - Only saves signals that have a signal peak higher than the cutoff value <br>
SNRChannelConsolidator.py - Takes SNR files sorted by channel and combines them <br>
TraceCounter.py - Prints number of traces in a file <br>
TraceShifter.py - Used to shift traces after the dataset has already been created <br>
<br>
The `data` folder contains all data from `data-production`.
