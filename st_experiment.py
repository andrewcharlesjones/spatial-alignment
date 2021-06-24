import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy
from autograd.scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_normal as mvno
import autograd.numpy as np
from autograd import grad, value_and_grad
from autograd.misc.optimizers import adam
from autograd.scipy.special import multigammaln
# from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import pandas as pd
inv = np.linalg.inv
from warp_gp_multigene import rbf_covariance, polar_warp, distance_matrix

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True

def unpack_kernel_params(params):
    noise_scales = np.exp(params[:n_noise_scale_params]) + 0.001
    cov_params  = params[n_noise_scale_params:]
    return cov_params, noise_scales

def unpack_params(params, n1, n2, n_kernel_params):
	n = n1 + n2
	kernel_params = params[:n_kernel_params]
	warped_coords = np.reshape(params[n_kernel_params:], (n, 2))
	X1_warped = warped_coords[:n1, :]
	X2_warped = warped_coords[n1:, :]
	return X1_warped, X2_warped, kernel_params


def get_coordinates(df):
    """
    Extracts spatial coordinates from ST data with index in 'AxB' type format.
    
    Return: pandas dataframe of coordinates
    """
    coor = []
    for spot in df.index:
        coordinates = spot.split('x')
        coordinates = [float(i) for i in coordinates]
        coor.append(coordinates)
    return np.array(coor)

def gp_likelihood(params):

	X1_warped, X2_warped, kernel_params = unpack_params(params, n1, n2, n_kernel_params)
	cov_params, noise_scales = unpack_kernel_params(kernel_params)
	noise_scale_obs, noise_scale_warp = noise_scales
	cov_params_obs, cov_params_warp = cov_params[:n_cov_params], cov_params[n_cov_params:]
	X_warped = np.vstack([X1_warped, X2_warped])

	# Compute log likelihood
	mean = np.zeros(n)

	# Form required covariance matrices
	covariance_obs = kernel(X_warped, X_warped, cov_params_obs) + noise_scale_obs * np.eye(n)
	covariance_warp = kernel(X_observed, X_observed, cov_params_warp) + noise_scale_warp * np.eye(n)
	# print(noise_scale_warp)

	# Warp log likelihood
	# import ipdb; ipdb.set_trace()
	LL_warp = np.sum([mvn.logpdf(X_warped[:, jj], mean, covariance_warp) for jj in range(2)])

	# Observation log likelihood
	LL_obs = np.sum([mvn.logpdf(Y[:, jj], mean, covariance_obs) for jj in range(n_genes)])
	
	# Penalty for preserving pairwise distances between points
	distance_mat_X1_warped = distance_matrix(X1_warped, X1_warped)
	distance_mat_X2_warped = distance_matrix(X2_warped, X2_warped)
	X1_penalty = np.sum((distance_mat_X1_warped - distance_mat_X1)**2)
	X2_penalty = np.sum((distance_mat_X2_warped - distance_mat_X2)**2)
	distance_penalty = X1_penalty + X2_penalty

	return -LL_warp - LL_obs #+ 1e0 * distance_penalty


if __name__ == "__main__":
	layer2_path = "./data/st/layer2.csv"
	layer2_raw_df = pd.read_csv(layer2_path, index_col=0)
	X_orig = get_coordinates(layer2_raw_df)

	n, n_genes_total = layer2_raw_df.shape
	# sigma2 = 1

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
	n1 = n // 2
	n2 = n - n1
	data1_idx = np.random.choice(np.arange(n), size=n1, replace=False)
	data2_idx = np.random.choice(np.arange(n), size=n2, replace=False)
	# data2_idx = np.setdiff1d(np.arange(n), data1_idx)
	# assert len(np.intersect1d(data1_idx, data2_idx)) == 0

	X1 = X_orig[data1_idx, :]
	X2 = X_orig[data2_idx, :]

	Y1 = Y_orig[data1_idx, :]
	Y2 = Y_orig[data2_idx, :]
	Y = np.concatenate([Y1, Y2], axis=0)

	## Warp coordinates
	linear_coeffs = np.random.normal(scale=0.01, size=n_views * 2 * 2)
	r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
	r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

	# r2s_true = np.ones(n2)
	# theta2s_true = np.ones(n2) + np.pi / 3
	X1_observed = polar_warp(X1, r1s_true, theta1s_true)
	X2_observed = polar_warp(X2, r2s_true, theta2s_true)
	X_observed = np.vstack([X1_observed, X2_observed])

	distance_mat_true = distance_matrix(X_orig, X_orig)
	distance_mat_X1 = distance_matrix(X1_observed, X1_observed)
	distance_mat_X2 = distance_matrix(X2_observed, X2_observed)

	### Fit model

	LL_grad = grad(gp_likelihood)

	fig = plt.figure(figsize=(14, 7), facecolor='white')
	data_ax = fig.add_subplot(121, frameon=False)
	aligned_ax = fig.add_subplot(122, frameon=False)
	plt.show(block=False)

	def summary(pars):
		

		X1_warped, X2_warped, kernel_params = unpack_params(pars, n1, n2, n_kernel_params)
		cov_params, noise_scale = unpack_kernel_params(kernel_params)

		X_warped = np.vstack([X1_warped, X2_warped])

		X1_penalty = np.sum(np.diag(distance_matrix(X1_warped, X2_warped)))

		print('LL {0:1.3e}'.format(gp_likelihood(pars)))

		data_ax.cla()
		aligned_ax.cla()

		data_ax.scatter(X1_observed[:, 0], X1_observed[:, 1], c=Y1[:, 0], label="X1")
		data_ax.scatter(X2_observed[:, 0], X2_observed[:, 1], c=Y2[:, 0], marker="+", label="X2")
		data_ax.legend()

		aligned_ax.scatter(X1_warped[:, 0], X1_warped[:, 1], c=Y1[:, 0], label="X1")
		aligned_ax.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2[:, 0], marker="+", label="X2")
		plt.legend()
		plt.draw()
		plt.pause(1.0/60.0)


	# "Parameters" are the warped coordinates
	n_cov_params = 2
	n_noise_scale_params = 2
	n_kernel_params = n_cov_params * 2 + n_noise_scale_params
	param_init = np.concatenate([
		np.ones(n_kernel_params),
		np.ndarray.flatten(X_observed)
		])

	X1_warped, X2_warped, kernel_params = unpack_params(param_init, n1, n2, n_kernel_params)
	cov_params, noise_scale = unpack_kernel_params(kernel_params)
	X_warped = np.vstack([X1_warped, X2_warped])

	data_ax.cla()
	aligned_ax.cla()

	data_ax.scatter(X1_observed[:, 0], X1_observed[:, 1], c=Y1[:, 0], label="X1")
	data_ax.scatter(X2_observed[:, 0], X2_observed[:, 1], c=Y2[:, 0], marker="+", label="X2")
	data_ax.legend()

	aligned_ax.scatter(X1_warped[:, 0], X1_warped[:, 1], c=Y1[:, 0], label="X1")
	aligned_ax.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2[:, 0], marker="+", label="X2")
	plt.legend()
	plt.draw()
	plt.pause(30.0/60.0)

	res = minimize(value_and_grad(gp_likelihood), param_init, jac=True, method='CG', callback=summary)
	pars = res.x

	import ipdb; ipdb.set_trace()

