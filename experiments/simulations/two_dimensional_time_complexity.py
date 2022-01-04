import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
import time

sys.path.append("../..")
# from models.gpsa_vi_lmc import VariationalWarpGP
# from util import matern12_kernel, rbf_kernel
from gpsa import VariationalGPSA, matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod


sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data
# from plotting.callbacks import callback_twod
from util import ConvergenceChecker

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../paste")
from src.paste import PASTE, visualization


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 35

n_spatial_dims = 2
n_views = 2
# n_outputs = 10
m_G = 50
m_X_per_view = 50

N_EPOCHS = 10000
PRINT_EVERY = 100
# N_LATENT_GPS = 1


def two_d_gpsa(
	n_outputs,
	n_samples,
	n_epochs,
	n_latent_gps,
	warp_kernel_variance=0.1,
	noise_variance=0.0,
	plot_intermediate=True,
	fixed_view_idx=None,
):

	X, Y, n_samples_list, view_idx = generate_twod_data(
		n_views,
		n_outputs,
		grid_size=np.sqrt(n_samples).astype(int),
		n_latent_gps=n_latent_gps["expression"],
		kernel_lengthscale=5.0,
		kernel_variance=warp_kernel_variance,
		noise_variance=noise_variance,
	)
	n_samples_per_view = X.shape[0] // n_views

	##  PASTE
	slice1 = anndata.AnnData(np.exp(Y[view_idx[0]]))
	slice2 = anndata.AnnData(np.exp(Y[view_idx[1]]))

	slice1.obsm["spatial"] = X[view_idx[0]]
	slice2.obsm["spatial"] = X[view_idx[1]]

	start = time.time()
	pi12 = PASTE.pairwise_align(slice1, slice2, alpha=0.1)

	slices = [slice1, slice2]
	pis = [pi12]
	new_slices = visualization.stack_slices_pairwise(slices, pis)
	end = time.time()
	time_paste = end - start

	x = torch.from_numpy(X).float().clone()
	y = torch.from_numpy(Y).float().clone()

	data_dict = {
		"expression": {
			"spatial_coords": x,
			"outputs": y,
			"n_samples_list": n_samples_list,
		}
	}

	start = time.time()
	model = VariationalGPSA(
		data_dict,
		n_spatial_dims=n_spatial_dims,
		m_X_per_view=m_X_per_view,
		m_G=m_G,
		data_init=True,
		minmax_init=False,
		grid_init=False,
		n_latent_gps=n_latent_gps,
		# n_latent_gps=None,
		mean_function="identity_fixed",
		kernel_func_warp=rbf_kernel,
		kernel_func_data=rbf_kernel,
		# fixed_warp_kernel_variances=np.ones(n_views) * 1.0,
		# fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
		fixed_view_idx=fixed_view_idx,
	).to(device)

	view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

	def train(model, loss_fn, optimizer):
		model.train()

		# Forward pass
		G_means, G_samples, F_latent_samples, F_samples = model.forward(
			{"expression": x}, view_idx=view_idx, Ns=Ns, S=5
		)

		# Compute loss
		loss = loss_fn(data_dict, F_samples)

		# Compute gradients and take optimizer step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		return loss.item()

	# Set up figure.
	fig = plt.figure(figsize=(14, 7), facecolor="white", constrained_layout=True)
	data_expression_ax = fig.add_subplot(122, frameon=False)
	latent_expression_ax = fig.add_subplot(121, frameon=False)
	plt.show(block=False)

	window_size = 10
	threshold = 1e-2

	loss_trace = []
	error_trace = []
	decrease_in_loss = np.zeros(n_epochs)
	average_decrease_in_loss = np.zeros(n_epochs)
	for t in range(n_epochs):
		loss = train(model, model.loss_fn, optimizer)
		loss_trace.append(loss)

		if t >= 1:
			decrease_in_loss[t] = loss_trace[t-1] - loss_trace[t]
		if t >= window_size:
			average_decrease_in_loss[t] = np.mean(decrease_in_loss[t - window_size + 1:t])
			has_converged = (average_decrease_in_loss[t] < threshold)
			if has_converged:
				break

		if plot_intermediate and t % PRINT_EVERY == 0:
			print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
			G_means, G_samples, F_latent_samples, F_samples = model.forward(
				{"expression": x}, view_idx=view_idx, Ns=Ns
			)

		G_means, G_samples, F_latent_samples, F_samples = model.forward(
			{"expression": x}, view_idx=view_idx, Ns=Ns
		)

	end = time.time()
	time_gpsa = end - start
	print("Done!")


	plt.close()

	return time_paste, time_gpsa


if __name__ == "__main__":

	n_repeats = 3
	n_outputs = 30
	n_samples_list = [64, 500, 1000]
	times_paste_array = np.zeros((n_repeats, len(n_samples_list)))
	times_gpsa_array = np.zeros((n_repeats, len(n_samples_list)))
	for ii in range(n_repeats):
		for jj, n_samples in enumerate(n_samples_list):
			time_paste, time_gpsa = two_d_gpsa(
				n_epochs=N_EPOCHS,
				n_samples=n_samples,
				n_outputs=n_outputs,
				warp_kernel_variance=0.5,
				noise_variance=0.001,
				n_latent_gps={"expression": 5},
				fixed_view_idx=None,
			)
			times_paste_array[ii, jj] = time_paste
			times_gpsa_array[ii, jj] = time_gpsa

		times_paste_df = pd.melt(pd.DataFrame(times_paste_array, columns=n_samples_list))
		times_paste_df["method"] = "PASTE"
		times_gpsa_df = pd.melt(pd.DataFrame(times_gpsa_array, columns=n_samples_list))
		times_gpsa_df["method"] = "GPSA"

		times_df = pd.concat([times_paste_df, times_gpsa_df], axis=0)
		times_df.to_csv("./out/time_experiment_results.csv", index=False)
	

	import matplotlib

	font = {"size": LATEX_FONTSIZE}
	matplotlib.rc("font", **font)
	matplotlib.rcParams["text.usetex"] = True

	sns.lineplot(data=times_df, x="variable", y="value", hue="method")
	plt.show()
	import ipdb; ipdb.set_trace()

