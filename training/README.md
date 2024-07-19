This folder contains all scripts that pertain to training models or that are specifically written for the classifier or denoiser.

AnalyzeClassifierAndDenoiser.ipynb - Jupyter Notebook used to plot the results of the networks <br>
Training_2D_1D_Networks.ipynb - Jupyter notebook used to train both the classifier and denoiser
<ul>
  <li>This script has produced better results than other network-training files in this repository</li>
  <li>The script is based on the script with the same name in <a href="https://github.com/AbdulRehmanUDEL/PhdAnalysis">Abdul Rehman's PhD repository</a></li>
</ul>
<br>

Note: you will need to create directories to save models, loss/accuracy data, and plots. All network scripts follow the same naming convention for saving files. An example is below. <br>
`
ModelsDir = ABS_PATH_HERE + f"/data/models/Denoiser_MSE_Updated_{dataset}"
LossAccDir = ABS_PATH_HERE + f"/data/loss_and_accuracy/Denoiser_MSE_Updated_{dataset}"
PlotsDir = ABS_PATH_HERE + f"/plots/Denoiser_MSE_Updated_{dataset}_Plots"
`
