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


if __name__ == "__main__":
	layer2_path = "./data/st/layer2.csv"
	layer2_raw_df = pd.read_csv(layer2_path, index_col=0)
	X_orig = get_st_coordinates(layer2_raw_df)
	X_orig -= X_orig.min(0)
	X_orig /= X_orig.max(0)

	n, n_genes_total = layer2_raw_df.shape

	kernel = rbf_covariance

	## Select high-variance genes
	n_genes = 10
	gene_vars = layer2_raw_df.var(0).values
	sorted_idx = np.argsort(-gene_vars)
	chosen_idx = sorted_idx[:n_genes]
	gene_names = layer2_raw_df.columns.values[chosen_idx]
	Y_orig_unstdized = np.log(layer2_raw_df.values[:, chosen_idx] + 1)

	assert X_orig.shape[0] == Y_orig_unstdized.shape[0]

	## Standardize expression
	Y_orig = (Y_orig_unstdized - Y_orig_unstdized.mean(0)) / Y_orig_unstdized.std(0)

	## Artificially split into two datasets
	n_views = 2
	# n1 = n // 2
	# n2 = n - n1
	n1, n2 = n, n
	n_total = n1 + n2
	p = 2
	data1_idx = np.random.choice(np.arange(n), size=n1, replace=False)
	data2_idx = np.random.choice(np.arange(n), size=n2, replace=False)
	view_idx = np.array([np.arange(0, n1), np.arange(n1, n1 + n2)])

	X1 = X_orig
	X2 = X_orig
	Y1 = Y_orig
	Y2 = Y_orig
	# X1 = X_orig[data1_idx, :]
	# X2 = X_orig[data2_idx, :]
	# Y1 = Y_orig[data1_idx, :]
	# Y2 = Y_orig[data2_idx, :]
	Y = np.concatenate([Y1, Y2], axis=0)


	## Warp coordinates
	linear_coeffs = np.random.normal(scale=0.01, size=n_views * 2 * 2)
	r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
	r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

	X1_observed = polar_warp(X1, r1s_true, theta1s_true)
	X2_observed = polar_warp(X2, r2s_true, theta2s_true)
	X = np.vstack([X1_observed, X2_observed])

	warp_gp = TwoLayerWarpGP(
		X, Y, n_views=n_views, n_samples_list=[n1, n2], kernel=rbf_covariance
	)
	warp_gp.fit(plot_updates=True, print_every=10)


	import ipdb; ipdb.set_trace()

