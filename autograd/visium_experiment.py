import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy
from autograd.scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_normal as mvno
import autograd.numpy as np
from autograd import grad, value_and_grad
from autograd.misc.optimizers import adam
import pandas as pd
from gp_functions import rbf_covariance
from warp_gp import TwoLayerWarpGP
from util import get_st_coordinates, polar_warp

import matplotlib
font = {"size": 15}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True

N_SAMPLES = 30


if __name__ == "__main__":
	spatial_locs_sample1_path = "data/visium/mouse_brain/sample1/filtered_feature_bc_matrix/spatial_locs_small.csv"
	data_sample1_path = "data/visium/mouse_brain/sample1/filtered_feature_bc_matrix/gene_expression_small.csv"
	X_df_sample1 = pd.read_csv(spatial_locs_sample1_path, index_col=0)
	Y_df_sample1 = pd.read_csv(data_sample1_path, index_col=0)
	# X_df_sample1, Y_df_sample1 = X_df_sample1.iloc[:N_SAMPLES, :], Y_df_sample1.iloc[:N_SAMPLES, :]


	spatial_locs_sample2_path = "data/visium/mouse_brain/sample2/filtered_feature_bc_matrix/spatial_locs_small.csv"
	data_sample2_path = "data/visium/mouse_brain/sample2/filtered_feature_bc_matrix/gene_expression_small.csv"
	X_df_sample2 = pd.read_csv(spatial_locs_sample2_path, index_col=0)
	Y_df_sample2 = pd.read_csv(data_sample2_path, index_col=0)
	# X_df_sample2, Y_df_sample2 = X_df_sample2.iloc[:N_SAMPLES, :], Y_df_sample2.iloc[:N_SAMPLES, :]

	X_orig_sample1 = X_df_sample1[["array_col", "array_row"]]
	X_orig_sample2 = X_df_sample2[["array_col", "array_row"]]

	X_orig_sample1 -= X_orig_sample1.min(0)
	X_orig_sample1 /= X_orig_sample1.max(0)
	X_orig_sample2 -= X_orig_sample2.min(0)
	X_orig_sample2 /= X_orig_sample2.max(0)
	X_orig_sample1 = X_orig_sample1.values
	X_orig_sample2 = X_orig_sample2.values

	assert np.all(X_df_sample1.index.values == Y_df_sample1.index.values)
	assert np.all(X_df_sample2.index.values == Y_df_sample2.index.values)

	n, n_genes_total = Y_df_sample1.shape

	kernel = rbf_covariance

	## Select high-variance genes
	n_genes = 100
	chosen_idx = np.arange(n_genes)
	gene_names = Y_df_sample1.columns.values[chosen_idx]
	Y_orig_unstdized_sample1 = Y_df_sample1.values[:, chosen_idx]
	Y_orig_unstdized_sample2 = Y_df_sample2.values[:, chosen_idx]

	assert X_orig_sample1.shape[0] == Y_orig_unstdized_sample1.shape[0]
	assert X_orig_sample2.shape[0] == Y_orig_unstdized_sample2.shape[0]

	## Standardize expression
	Y1 = (Y_orig_unstdized_sample1 - Y_orig_unstdized_sample1.mean(0)) / Y_orig_unstdized_sample1.std(0)
	Y2 = (Y_orig_unstdized_sample2 - Y_orig_unstdized_sample2.mean(0)) / Y_orig_unstdized_sample2.std(0)

	

	n_views = 2
	n1, n2 = Y1.shape[0], Y2.shape[0]
	n_total = n1 + n2
	p = 2
	data1_idx = np.random.choice(np.arange(n), size=n1, replace=False)
	data2_idx = np.random.choice(np.arange(n), size=n2, replace=False)
	view_idx = np.array([np.arange(0, n1), np.arange(n1, n1 + n2)])

	X1 = X_orig_sample1
	X2 = X_orig_sample2
	Y = np.concatenate([Y1, Y2], axis=0)

	plt.figure(figsize=(21, 7))
	plt.subplot(131)
	plt.scatter(X1[:, 0], X1[:, 1], marker=".", c=Y1[:, 5], label="Sample 1", s=200, alpha=0.3)
	plt.subplot(132)
	plt.scatter(X2[:, 0], X2[:, 1], marker=".", c=Y2[:, 5], label="Sample 2", s=200, alpha=0.3)
	plt.subplot(133)
	plt.scatter(X1[:, 0], X1[:, 1], marker=".", c=Y1[:, 5], label="Sample 1", s=200, alpha=0.3)
	plt.scatter(X2[:, 0], X2[:, 1], marker="+", c=Y2[:, 5], label="Sample 2", s=200, alpha=0.3)
	plt.show()
	import ipdb; ipdb.set_trace()


	## Warp coordinates
	# linear_coeffs = np.random.normal(scale=0.01, size=n_views * 2 * 2)
	# r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
	# r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

	# X1_observed = polar_warp(X1, r1s_true, theta1s_true)
	# X2_observed = polar_warp(X2, r2s_true, theta2s_true)
	X = np.vstack([X1, X2])

	warp_gp = TwoLayerWarpGP(
		X, Y, n_views=n_views, n_samples_list=[n1, n2], kernel=rbf_covariance
	)
	warp_gp.fit(plot_updates=True, print_every=10)


	import ipdb; ipdb.set_trace()

