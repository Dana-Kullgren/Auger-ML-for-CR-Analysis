#!/bin/bash

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`

HERE=$(dirname $(realpath -s $0))
# BACKGROUND=/mnt/ceph1-npx/user/dkullgren/WaveformML-DataOnly/data-production/taxi-noise/data/AugerBackground/i3Files
BACKGROUND=/data/user/sverpoest/radio/Auger/data/raw
SIMDIR=/mnt/ceph1-npx/user/dkullgren/WaveformML-DataOnly/data-production/taxi-noise/data/AugerSims/discrete/proton

PYTHONSCP=$HERE/MakeTracesAuger.py

#INPUT=$(ls -d $SIMDIR/lgE_17.0/sin2_*/?????${1})

INPUT=""
# for ZEN in {8..4}; do
# for ZEN in {9..5}; do
# for ZEN in 0 10 20 30 40; do
for ZEN in 40 30 20 10 0; do
  # INPUT="$INPUT $(ls -d $SIMDIR/lgE_17.0/sin2_0.${ZEN}/?????${1} $SIMDIR/lgE_17.1/sin2_0.${ZEN}/?????${1})"
  # INPUT="$INPUT $(ls -d $SIMDIR/lgE_17.5/sin2_0.${ZEN}/?????${1} $SIMDIR/lgE_17.6/sin2_0.${ZEN}/?????${1} $SIMDIR/lgE_17.7/sin2_0.${ZEN}/?????${1} $SIMDIR/lgE_17.8/sin2_0.${ZEN}/?????${1} $SIMDIR/lgE_17.9/sin2_0.${ZEN}/?????${1} $SIMDIR/lgE_18.0/sin2_0.${ZEN}/?????${1})"
  # INPUT="$INPUT $(ls -d $SIMDIR/lgE_17.0/Zen_${ZEN}/?????${1} $SIMDIR/lgE_17.2/Zen_${ZEN}/?????${1} $SIMDIR/lgE_17.5/Zen_${ZEN}/?????${1} $SIMDIR/lgE_17.8/Zen_${ZEN}/?????${1} $SIMDIR/lgE_18.0/Zen_${ZEN}/?????${1} $SIMDIR/lgE_18.2/Zen_${ZEN}/?????${1})"
  INPUT="$INPUT $(ls -d $SIMDIR/lgE_16.5/Zen_${ZEN}/?????${1} $SIMDIR/lgE_16.8/Zen_${ZEN}/?????${1} $SIMDIR/lgE_17.0/Zen_${ZEN}/?????${1} $SIMDIR/lgE_17.2/Zen_${ZEN}/?????${1} $SIMDIR/lgE_17.5/Zen_${ZEN}/?????${1} $SIMDIR/lgE_17.8/Zen_${ZEN}/?????${1} $SIMDIR/lgE_18.0/Zen_${ZEN}/?????${1}  $SIMDIR/lgE_18.2/Zen_${ZEN}/?????${1})"
done

echo "BACKGROUND: $BACKGROUND"
echo "PYTHONSCP: $PYTHONSCP"
echo "Input 1: $1"

# TAXIFILES=$(ls $BACKGROUND/eventData_*2023-05-*${1}.i3.gz)
TAXIFILES=$(ls $BACKGROUND/eventData_*2023-11-*${1}.i3.gz $BACKGROUND/eventData_*2024-01-*${1}.i3.gz)
# TAXIFILES=$(ls $BACKGROUND/eventData_*202*${1}.i3.gz)
# TAXIFILES=$(ls $BACKGROUND/eventData_*2022-11-?${1}*.i3.gz $BACKGROUND/eventData_*2022-12-?${1}*.i3.gz $BACKGROUND/eventData_*2023-01-?${1}*.i3.gz $BACKGROUND/eventData_*2023-03-?${1}*.i3.gz $BACKGROUND/eventData_*2023-04-?${1}*.i3.gz $BACKGROUND/eventData_*2023-05-?${1}*.i3.gz)
# TAXIFILES=$(ls $BACKGROUND/eventData_*2023-01-?${1}*.i3.gz $BACKGROUND/eventData_*2023-02-?${1}*.i3.gz $BACKGROUND/eventData_*2023-03-?${1}*.i3.gz $BACKGROUND/eventData_*2023-04-?${1}*.i3.gz)
# TAXIFILES=$(ls $TAXIDIR22/eventData_*2022-01-?${1}*.i3.gz $TAXIDIR22/eventData_*2022-02-?${1}*.i3.gz $TAXIDIR22/eventData_*2022-03-?${1}*.i3.gz $TAXIDIR22/eventData_*2022-04-?${1}*.i3.gz)
# TAXIFILES=$(ls $TAXIDIR20/eventData_*2020-12-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-01-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-02-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-03-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-10-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-11-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-12-?${1}*.i3.gz)
# TAXIFILES=$(ls $TAXIDIR21/eventData_*2021-04-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-05-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-06-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-07-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-08-?${1}*.i3.gz $TAXIDIR21/eventData_*2021-09-?${1}*.i3.gz)

echo "TAXIFILES:"

echo "$TAXIFILES"

ICETRAY_ENV=/home/dkullgren/surface-array/build/env-shell.sh
# ICETRAY_ENV=/data/user/arehman/Git/surface-array/build/env-shell.sh

$ICETRAY_ENV $PYTHONSCP --taxi $TAXIFILES --output Run${1} --input $INPUT
