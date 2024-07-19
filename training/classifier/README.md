Classifier.py - Creates classifier models (runs on combined channels) <br>
Classifier.sh, Classifier.sub - Bash and submission scripts to submit a job _to cobalt_ <br>
ClassifierJob.qs - Submission script to submit a job _to caviness_ <br>
Classifier2Channels.py - Creates classifier models (applies 1D convolution to two channels) <br>
LossMinClassifier.py - Plots the minimum loss of classifier models, splitting up models based on layer, kernel size, and filter number
<br>
<br>
Classifier models and loss/accuracy data are saved to the `data` folder and classifier plots are saved to the `plots` folder.
<br>
<br>
<br>
**Submitting classifier jobs (in caviness):**

To update classifier model settings (ks, layers, dl, fil), change the arguments given in ClassifierJob.qs as seen below.

Arguments example: `python /home/2332/work/WaveformML/training/classifier/Classifier.py --ks 32 --layers 2 3 4 --dl 1 --fil 64`
