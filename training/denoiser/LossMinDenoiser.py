import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os

ABS_PATH_HERE=str(os.path.dirname(os.path.realpath(__file__)))

# Enter directories here
LossAccDir = ABS_PATH_HERE + "/data/loss_and_accuracy/Denoiser_MSE_v6_Small"
PlotsDir = ABS_PATH_HERE + "/plots/MSE_Denoiser_v6_Plots_Small"


# NOTE: Only include the jobs that ran after we removed the NoiseOnly traces
FIL = [16, 32, 64]
KS = [128]
Layers = [2, 3]

# model_list = []
# loss_list = []

# models_2, models_3, models_4, models_5, models_6, models_7, models_8 = [], [], [], [], [], [], []
# model_layer_lists = [models_2, models_3, models_4, models_5, models_6, models_7, models_8]

# loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8 = [], [], [], [], [], [], []
# loss_layer_lists = [loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8]

models_layer_2, models_layer_3, models_layer_4, models_layer_5, models_layer_6, models_layer_7, models_layer_8 = [], [], [], [], [], [], []
model_layer_lists = [models_layer_2, models_layer_3, models_layer_4, models_layer_5, models_layer_6, models_layer_7, models_layer_8]

models_ks_2, models_ks_3, models_ks_4, models_ks_5, models_ks_6, models_ks_7, models_ks_8 = [], [], [], [], [], [], []
model_ks_lists = [models_ks_2, models_ks_3, models_ks_4, models_ks_5, models_ks_6, models_ks_7, models_ks_8]

models_fil_2, models_fil_3, models_fil_4, models_fil_5, models_fil_6, models_fil_7, models_fil_8 = [], [], [], [], [], [], []
model_fil_lists = [models_fil_2, models_fil_3, models_fil_4, models_fil_5, models_fil_6, models_fil_7, models_fil_8]

loss_layer_2, loss_layer_3, loss_layer_4, loss_layer_5, loss_layer_6, loss_layer_7, loss_layer_8 = [], [], [], [], [], [], []
loss_layer_lists = [loss_layer_2, loss_layer_3, loss_layer_4, loss_layer_5, loss_layer_6, loss_layer_7, loss_layer_8]

loss_ks_2, loss_ks_3, loss_ks_4, loss_ks_5, loss_ks_6, loss_ks_7, loss_ks_8 = [], [], [], [], [], [], []
loss_ks_lists = [loss_ks_2, loss_ks_3, loss_ks_4, loss_ks_5, loss_ks_6, loss_ks_7, loss_ks_8]

loss_fil_2, loss_fil_3, loss_fil_4, loss_fil_5, loss_fil_6, loss_fil_7, loss_fil_8 = [], [], [], [], [], [], []
loss_fil_lists = [loss_fil_2, loss_fil_3, loss_fil_4, loss_fil_5, loss_fil_6, loss_fil_7, loss_fil_8]

ymax = .0035

# fig_length = len(model_list)
NRows = 1
NCols = 1
gs = gridspec.GridSpec(NRows, NCols, wspace=0.05, hspace=0.3)
fig = plt.figure(figsize=(25*NCols, 5*NRows))
fig.suptitle('Mimimum Loss of Denoiser Models (Layers=2, KS=128)')

# layer_idx = 0
# ks_idx = 1
fil_idx = 0

# for i in range(len(Layers)):
# 	for ks in KS:
# 		for fil in FIL:
# 			if (Layers[i] == 2 or Layers[i] == 3) and fil == 64:
# 				continue
# 			model_layer_lists[i].append('Fil={0} \n KS={1}'.format(fil, ks))
# 			loss = np.load(LossAccDir + '/TstLoss_Fil={0}_KS={1}_Layers={2}.npy'.format(fil, ks, Layers[i]))
# 			min_loss = np.min(loss)
# 			loss_layer_lists[i].append(min_loss)
# 			# acc = np.load(LossAccDir + '/TstAcc_Layers={0}_DL={1}_Fil={2}_KS={3}.npy'.format(Layers[i], DL, fil, ks))
# 			# max_acc = np.max(acc)
# 			# acc_layer_lists[i].append(max_acc)

# 	print(model_layer_lists[i])
# 	print(loss_layer_lists[i])

# 	x = model_layer_lists[i]
# 	y_loss = loss_layer_lists[i]
# 	# y_acc = acc_layer_lists[i]

# 	ax = fig.add_subplot(gs[layer_idx])
# 	ax.plot(x, y_loss)
# 	ax. set_ylim(top=ymax)
# 	ax.set_xlabel('Models')
# 	ax.set_ylabel('Minimum Testing Loss')
# 	ax.set_title('Layers = ' + str(Layers[i]))

# 	layer_idx += NCols

# for i in range(len(KS)):
# 	for layer in Layers:
# 		for fil in FIL:
# 			if (layer == 2 or layer == 3) and fil == 64:
# 				continue
# 			model_ks_lists[i].append('Fil={0} \n Layers={1}'.format(fil, layer))
# 			loss = np.load(LossAccDir + '/TstLoss_Fil={0}_KS={1}_Layers={2}.npy'.format(fil, KS[i], layer))
# 			min_loss = np.min(loss)
# 			loss_ks_lists[i].append(min_loss)
# 			# acc = np.load(LossAccDir + '/TstAcc_Layers={0}_DL={1}_Fil={2}_KS={3}.npy'.format(layer, DL, fil, KS[i]))
# 			# max_acc = np.max(acc)
# 			# acc_layer_lists[i].append(max_acc)

# 	x = model_ks_lists[i]
# 	y_loss = loss_ks_lists[i]
# 	# y_acc = acc_ks_lists[i]

# 	ax = fig.add_subplot(gs[ks_idx])
# 	ax.plot(x, y_loss)
# 	ax. set_ylim(top=ymax)
# 	ax.set_xlabel('Models')
# 	ax.set_ylabel('Minimum Testing Loss')
# 	ax.set_title('KS = ' + str(KS[i]))

# 	ks_idx += 3

# for i in range(len(FIL)):
# 	for layer in Layers:
# 		for ks in KS:
# 			if (layer == 2 or layer == 3) and FIL[i] == 64:
# 				continue
# 			model_fil_lists[i].append('Layers={0} \n KS={1}'.format(layer, ks))
# 			loss = np.load(LossAccDir + '/TstLoss_Fil={0}_KS={1}_Layers={2}.npy'.format(FIL[i], ks, layer))
# 			min_loss = np.min(loss)
# 			loss_fil_lists[i].append(min_loss)
# 			# acc = np.load(LossAccDir + '/TstAcc_Layers={0}_DL={1}_Fil={2}_KS={3}.npy'.format(layer, DL, FIL[i], ks))
# 			# max_acc = np.max(acc)
# 			# acc_layer_lists[i].append(max_acc)

# 	x = model_fil_lists[i]
# 	y_loss = loss_fil_lists[i]
# 	# y_acc = acc_fil_lists[i]


# 	ax = fig.add_subplot(gs[fil_idx])
# 	ax.plot(x, y_loss)
# 	ax.set_xlabel('Models')
# 	ax.set_ylabel('Minimum Testing Loss')
# 	ax. set_ylim(top=ymax)
# 	ax.set_title('FIL = ' + str(FIL[i]))

# 	fil_idx += NCols


fil_model_list = []
fil_loss_list = []

for fil in FIL:
	for layer in Layers:
		for ks in KS:
			if (fil==16) or (layer==2):
				fil_model_list.append('FIL={0}_Layers={1}'.format(fil, layer))
				loss = np.load(LossAccDir + '/TstLoss_Fil={0}_KS={1}_Layers={2}.npy'.format(fil, ks, layer))
				min_loss = np.min(loss)
				fil_loss_list.append(min_loss)

# print(fil_model_list)
print(fil_loss_list)

x = fil_model_list
y_loss = fil_loss_list

ax = fig.add_subplot(gs[0])
ax.plot(x, y_loss)
ax.set_xlabel('Models')
ax.set_ylabel('Minimum Testing Loss')
# ax.set_title('FIL = ' + str(FIL[i]))

fig.savefig(PlotsDir + '/MinLossOfDenoiserModels.pdf', bbox_inches='tight')