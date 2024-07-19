#!/bin/bash -l
#

#SBATCH --export=NONE
#SBATCH --partition=standard
#SBATCH --time=2-0
#SBATCH --job-name="DenoiserJob2D-ant1"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100

# use node r06g05 if job >> 5 min

#UD_QUIET_JOB_SETUP=YES
# SBATCH --output=/work/icecube/users/arehman/Tunka/denoiser/Output/output.txt


vpkg_devrequire anaconda/5.2.0:python3
source activate tf-gpu
#
# Do general job environment setup:
#
. /opt/shared/slurm/templates/libexec/common.sh

#
# [EDIT] Insert your job script commands here...
# python  $CNNBASHHOME/training/denoiser/Denoiser.py
python /home/2332/work/WaveformML/training/denoiser/Denoiser2D.py --fil 6 --ks 140 --layers 3 --ant ant1

rc=$?

exit $rc

#
