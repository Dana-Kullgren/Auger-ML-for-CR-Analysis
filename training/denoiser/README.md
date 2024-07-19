Denoiser.py - Creates denoiser models (runs on combined channels) <br>
Denoiser.sh, Denoiser.sub - Bash and submission scripts to submit a job _to cobalt_ <br>
Denoiser2Channels.py - Creates denoiser models (applies 1D convolution to two channels) <br>
DenoiserJob.qs - Submission script to submit a job _to caviness_ <br>
LossMinDenoiser.py - Plots the minimum loss of denoiser models, splitting up models based on layer, kernel size, and filter number <br>
<br>
<br>
Denoiser models and loss/accuracy data are saved to the `data` folder and denoiser plots are saved to the `plots` folder.
<br>
<br>
<br>
**Submitting denoiser jobs (in cobalt):**

To update denoiser model settings (fil, ks, layers), change the arguments given in Denoiser.sub. A model will be trained based on every combination of arguments given.

Arguments example: `arguments = --fil 20 --ks 8 16 32 64 128 256 512 --layers 3`

To submit a new job, change the names of the error, log, and output files in Denoiser.sub based on the name of the job to be submitted (ex: DenoiserJob1.out).
  
To set the name of the job when submitting, use: `condor_submit -batch-name DenoiserJob[job_number] Denoiser.sub`
