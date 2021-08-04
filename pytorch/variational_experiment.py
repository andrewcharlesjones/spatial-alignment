import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.contrib.gp as gp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from util import make_pinwheel
import seaborn as sns
from variational_warp_gp import VariationalWarpGP
import sys
sys.path.append("..")
from gp_functions import rbf_covariance
from scipy.stats import multivariate_normal as mvnpy
from util import polar_warp

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

	# n = 30
	# n_spatial_dims = 1
	# n_genes = 5
	# m_X_per_view = 10
	# m_G = 15
	# n_views = 2
	# noise_variance = 0.01
	# X_distribution = torch.distributions.Uniform(low=-10, high=10)
	# X_orig = X_distribution.sample(sample_shape=torch.Size([n, n_spatial_dims]))
	# X1 = X_orig.clone()
	# X2 = X_orig + 0.1 * torch.randn([n, n_spatial_dims])

	# kernel = gp.kernels.RBF(
	# 	input_dim=n_spatial_dims, variance=torch.tensor(1.0), lengthscale=torch.tensor(3.0)
	# )
	# Kxx = kernel(X_orig, X_orig)
	# Kxx_plus_noise = Kxx + noise_variance * torch.eye(n)

	# Y_distribution = torch.distributions.MultivariateNormal(
	# 	loc=torch.zeros(n), covariance_matrix=Kxx_plus_noise
	# )

	# Y = torch.cat(
	# 	[Y_distribution.sample(sample_shape=[1]) for _ in range(n_genes)], dim=0
	# ).t()
	# Y1 = Y.clone()
	# Y2 = Y.clone()

	# X = torch.cat([X1, X2])
	# Y = torch.cat([Y1, Y2])
	# view_idx = np.array([np.arange(0, n), np.arange(n, 2 * n)])
	# assert len(view_idx) == n_views

	n_views = 2
	n_genes = 10
	kernel = rbf_covariance
	kernel_params_true = np.array([1.0, 1.0])
	m_X_per_view = 15
	m_G = 15
	# n_samples_per_view = 30

	xlimits = [-10, 10]
	ylimits = [-10, 10]
	numticks = 6
	x1s = np.linspace(*xlimits, num=numticks)
	x2s = np.linspace(*ylimits, num=numticks)
	X1, X2 = np.meshgrid(x1s, x2s)
	X_orig = np.vstack([X1.ravel(), X2.ravel()]).T
	n_samples_per_view = X_orig.shape[0]

	n_samples_list = [n_samples_per_view] * n_views
	cumulative_sums = np.cumsum(n_samples_list)
	cumulative_sums = np.insert(cumulative_sums, 0, 0)
	view_idx = np.array(
		[
			np.arange(cumulative_sums[ii], cumulative_sums[ii + 1])
			for ii in range(n_views)
		]
	)
	n = np.sum(n_samples_list)
	sigma2 = 1
	# X_orig = np.hstack(
	#     [
	#         np.random.uniform(low=-3, high=3, size=(n_samples_per_view, 1))
	#         for _ in range(2)
	#     ]
	# )
	Y_orig = np.vstack(
		[
			mvnpy.rvs(
				mean=np.zeros(n_samples_per_view),
				cov=kernel(X_orig, X_orig, kernel_params_true),
			)
			for _ in range(n_genes)
		]
	).T

	X = np.empty((np.sum(n_samples_list), 2))
	Y = np.empty((np.sum(n_samples_list), n_genes))

	for vv in range(n_views):

		curr_X = X_orig.copy()
		# Warp
		linear_coeffs = np.random.normal(scale=0.1, size=2 * 2)
		rs_true, thetas_true = curr_X @ linear_coeffs[:2], curr_X @ linear_coeffs[2:]

		curr_X_observed = polar_warp(curr_X, rs_true, thetas_true)
		X[view_idx[vv]] = curr_X_observed

		curr_Y = Y_orig.copy()
		Y[view_idx[vv]] = curr_Y  # + np.random.normal(scale=0.1, size=curr_Y.shape)
	
	x = torch.from_numpy(X).float().clone()
	y = torch.from_numpy(Y).float().clone()
	data_dict = {
		"expression": {
			"spatial_coords": x,
			"outputs": y,
			"n_samples_list": n_samples_list,
		}
	}

	# model = VGPR(X, view_idx, n, n_spatial_dims, m_X_per_view=m_X_per_view, m_G=m_G).to(device)
	model = VariationalWarpGP(data_dict, m_X_per_view=m_X_per_view, m_G=m_G).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

	def train(model, loss_fn, optimizer):
		model.train()

		# Forwrard pass
		G_samples, F_samples = model.forward(
			{"expression": x}
		)

		# Compute loss
		loss = loss_fn(data_dict, F_samples)

		# Compute gradients and take optimizer step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		return loss.item()

	# Set up figure.
	fig = plt.figure(figsize=(14, 7), facecolor="white")
	data_expression_ax = fig.add_subplot(121, frameon=False)
	latent_expression_ax = fig.add_subplot(122, frameon=False)
	plt.show(block=False)

	def callback(model, G_samples):
		model.eval()
		markers = [".", "+", "^"]

		data_expression_ax.cla()
		latent_expression_ax.cla()
		data_expression_ax.set_title("Expression data")
		latent_expression_ax.set_title("G, Expression")

		for vv in range(n_views):
			data_expression_ax.scatter(
				X[view_idx[vv], 0],
				X[view_idx[vv], 1],
				c=np.sum(Y[view_idx[vv], :], axis=1),
				label="View {}".format(vv + 1),
				marker=markers[vv],
				s=100,
			)
			# import ipdb; ipdb.set_trace()
			latent_expression_ax.scatter(
				model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
				model.G_means["expression"].detach().numpy()[view_idx[vv], 1],
				c=np.sum(Y[view_idx[vv], :], axis=1),
				label="View {}".format(vv + 1),
				marker=markers[vv],
				s=100,
			)
			latent_expression_ax.scatter(
				model.Xtilde[vv, :, 0].detach().numpy(),
				model.Xtilde[vv, :, 1].detach().numpy(),
				# c=np.sum(Y[view_idx[vv], :], axis=1),
				# label="View {}".format(vv + 1),
				marker=markers[vv],
				s=100,
			)
		plt.draw()
		plt.pause(1 / 60.0)

	N_EPOCHS = 5000
	PRINT_EVERY = 100
	loss_trace = []
	for t in range(N_EPOCHS):
		loss = train(model, model.loss_fn, optimizer)
		loss_trace.append(loss)
		if t % PRINT_EVERY == 0:
			print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
			G_samples, F_samples = model.forward(
				{"expression": x}
			)
			callback(model, G_samples)

	print("Done!")

	import ipdb

	ipdb.set_trace()