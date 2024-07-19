import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os

ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))
dataset = 'Auger_v4'
SeparatedAntennas = True
ant = 'ant1'
loss_function = 'MSE'

# Enter directories here
LossAccDir = ABS_PATH_HERE + f"/data/loss_and_accuracy/Classifier_{loss_function}_{dataset}"
PlotsDir = ABS_PATH_HERE + f"/plots/Classifier_{loss_function}_{dataset}_Plots"

# FIL = [8]
FIL = [4, 8, 16, 32, 64]
KS = [8, 16, 32, 64]
Layers = [2, 3, 4]
DLayers = [1, 2, 3]

model_list = []
loss_list = []
acc_list = []

NRows = len(KS)
NCols = len(Layers)
gs = gridspec.GridSpec(NRows, NCols, wspace=0.05, hspace=0.25)
fig = plt.figure(figsize=(5*NCols, 5*NRows))
fig.suptitle('Mimimum Loss of Classifier Models', size='xx-large')

gs_idx = 0
fil_model_list = []
plot_colors = ['b', 'g', 'r']
for ks in KS:
	for layer in Layers:
		ax = fig.add_subplot(gs[gs_idx])
		gs_idx += 1
		ax.set_title('KS = ' + str(ks) + ', Layers = ' + str(layer))
		for DL in DLayers:
			fil_model_list = []
			fil_loss_list = []
			fil_acc_list = []
			for fil in FIL:
				model_list.append('Layers={0} \n DL={1} \n Fil={2} \n KS={3}'.format(layer, DL, fil, ks))
				fil_model_list.append(f'Fil={fil}')
				if SeparatedAntennas:
					loss = np.load(LossAccDir + '/TstLoss_Layers={0}_DL={1}_Fil={2}_KS={3}_ANT={4}_lr=0.0001.npy'.format(layer, DL, fil, ks, ant))
					acc = np.load(LossAccDir + '/TstAcc_Layers={0}_DL={1}_Fil={2}_KS={3}_ANT={4}_lr=0.0001.npy'.format(layer, DL, fil, ks, ant))
				else:
					loss = np.load(LossAccDir + '/TstLoss_Layers={0}_DL={1}_Fil={2}_KS={3}.npy'.format(layer, DL, fil, ks))
					acc = np.load(LossAccDir + '/TstAcc_Layers={0}_DL={1}_Fil={2}_KS={3}.npy'.format(layer, DL, fil, ks))
				final_acc = acc[-1]
				final_loss = loss[-1]
				loss_list.append(final_loss)
				acc_list.append(final_acc)
				fil_loss_list.append(final_loss)
				fil_acc_list.append(final_acc)
				
			ax.plot(fil_model_list, fil_acc_list, color=plot_colors[DL-1], label=f'DL={DL}')
		ax.set_xlabel('Number of Filters')
		ax.set_ylabel('Minimum Testing Loss')
		ax.legend()

# x = model_list
# y_loss = loss_list

min_loss_idx = np.argmin(loss_list)
print("Best model: ", model_list[min_loss_idx])
print("Minimum loss: ", np.min(loss_list))
print("Maximum accuracy: ", np.max(acc_list))


fig.savefig(PlotsDir + '/MaxAccOfClassifierModels.png', bbox_inches='tight')