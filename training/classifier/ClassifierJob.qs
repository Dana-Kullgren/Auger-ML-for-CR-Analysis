#!/bin/bash -l
#

#SBATCH --export=NONE
#SBATCH --partition=standard
#SBATCH --time=2-0
#SBATCH --job-name="ClassifierJob"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100
#UD_QUIET_JOB_SETUP=YES


vpkg_devrequire anaconda/5.2.0:python3
source activate tf-gpu
#
# Do general job environment setup:
#
. /opt/shared/slurm/templates/libexec/common.sh

#
# [EDIT] Insert your job script commands here...
# python $CNNBASHHOME/training/classifier/Classifier.py
python /home/2332/work/WaveformML/training/classifier/Classifier2D.py --ks 160 --layers 2 --dl 1 --fil 12 --ant ant1

rc=$?

exit $rc

#
