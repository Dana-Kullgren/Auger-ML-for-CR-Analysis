#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE
# source the cvmfs
# eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/setup.sh`
# The cvmfs will be automatically loaded when you load the python environment
# source the virtual environment
source /home/arehman/work/TensorFlow/py3-v4.0.1_tensorflow2.3/bin/activate
$0=""
# python script to run
python /home/dkullgren/work/WaveformML/training/classifier/Classifier.py $@
