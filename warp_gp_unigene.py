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
inv = np.linalg.inv

import matplotlib
font = {"size": 20}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True


def polar_warp(X, r, theta):
	return np.array(
		[
			X[:, 0] + r * np.cos(theta),
			X[:, 1] + r * np.sin(theta)
		]).T

def distance_matrix(X, Y):
	return (np.expand_dims(X, 0) - np.expand_dims(Y, 1))**2

def gaussian_logpdf(x, mean, cov):
	p = x.shape[0]
	return -0.5 * (p * np.log(2. * np.pi) + np.linalg.slogdet(cov)[1] + (x - mean).T @ inv(cov) @ (x - mean))

def wishart_logpdf(x, V, n):
	p = x.shape[0]
	V_inv = inv(V)
	V_logdet = np.linalg.slogdet(V)[1]
	x_logdet = np.linalg.slogdet(x)[1]
	# import ipdb; ipdb.set_trace()
	return 0.5 * (n - p - 1) * x_logdet - 0.5 * np.trace(V_inv @ x) - 0.5 * n * p * np.log(2) - 0.5 * n * V_logdet - multigammaln(0.5 * p, p)

def gp_likelihood(params):

	X1_warped, X2_warped = unpack_params(params)
	X_warped = np.vstack([X1_warped, X2_warped])

	# Compute log likelihood
	mean = np.zeros(n)

	# Form required covariance matrices
	covariance_obs = kernel(X_warped, X_warped) + sigma2 * np.eye(n)

	# Warp log likelihood
	LL_warp = np.sum([mvn.logpdf(X_warped[:, jj], mean, covariance_warp) for jj in range(2)])

	# Observation log likelihood
	# LL_obs = gaussian_logpdf(Y, mean=mean, cov=covariance_obs)
	LL_obs = mvn.logpdf(Y, mean, covariance_obs)
	import ipdb; ipdb.set_trace()

	# Penalty for preserving pairwise distances between points
	distance_mat_X1_warped = distance_matrix(X1_warped, X1_warped)
	distance_mat_X2_warped = distance_matrix(X2_warped, X2_warped)
	X1_penalty = np.sum((distance_mat_X1_warped - distance_mat_X1)**2)
	X2_penalty = np.sum((distance_mat_X2_warped - distance_mat_X2)**2)
	distance_penalty = X1_penalty + X2_penalty

	# print(-LL_warp, -LL_obs, distance_penalty)

	return -LL_warp - LL_obs #+ 1e-5 * distance_penalty
	# return -LL_warp - LL_obs - L_prior
	# return distance_penalty

def rbf_covariance(x, xp):
	output_scale = 1 #np.exp(kernel_params[0])
	lengthscales = 1 #np.exp(kernel_params[1:])
	diffs = np.expand_dims(x / lengthscales, 1)\
		  - np.expand_dims(xp / lengthscales, 0)
	return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

# def rbf_covariance(kernel_params, x, xp):
#     output_scale = np.exp(kernel_params[0])
#     lengthscales = np.exp(kernel_params[1:])
#     diffs = np.expand_dims(x /lengthscales, 1)\
#           - np.expand_dims(xp/lengthscales, 0)
#     return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

def unpack_params(params):
	warped_coords = np.reshape(params, (n, 2))
	X1_warped = warped_coords[:n1, :]
	X2_warped = warped_coords[n1:, :]
	return X1_warped, X2_warped

if __name__ == "__main__":

	n_views = 2
	p = 2
	kernel = rbf_covariance
	n1, n2 = 100, 100
	n = n1 + n2
	sigma2 = 1
	x1, x2 = np.random.uniform(low=-3, high=3, size=(n1, 1)), np.random.uniform(low=-3, high=3, size=(n1, 1))
	X_orig = np.hstack([x1, x2])
	Y_orig = mvnpy.rvs(mean=np.zeros(n1), cov=kernel(X_orig, X_orig))

	X1 = X_orig.copy()
	Y1 = Y_orig.copy()

	X2 = X_orig.copy()
	Y2 = Y_orig.copy()

	X = np.vstack([X1, X2])
	Y = np.concatenate([Y1, Y2])

	# Warp
	linear_coeffs = np.random.normal(scale=0.1, size=n_views * p * 2)
	r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
	r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

	X1_observed = polar_warp(X1, r1s_true, theta1s_true)
	X2_observed = polar_warp(X2, r2s_true, theta2s_true)
	X_observed = np.vstack([X1_observed, X2_observed])

	covariance_observations = kernel(X_observed, X_observed)
	covariance_warp = covariance_observations + 0.1 * np.eye(n)
	covariance_warp_inv = inv(covariance_warp)

	x1, x2 = np.meshgrid(np.linspace(-10, 10, 5), np.linspace(-10, 10, 5))
	Xbar = np.vstack([x1.ravel(), x2.ravel()]).T

	distance_mat_true = distance_matrix(X_orig, X_orig)
	distance_mat_X1 = distance_matrix(X1_observed, X1_observed)
	distance_mat_X2 = distance_matrix(X2_observed, X2_observed)

	LL_grad = grad(gp_likelihood)

	fig = plt.figure(figsize=(14, 7), facecolor='white')
	data_ax = fig.add_subplot(121, frameon=False)
	aligned_ax = fig.add_subplot(122, frameon=False)
	plt.show(block=False)

	def summary(pars):
		print('LL {0:1.3e}'.format(gp_likelihood(pars)))

		X1_warped, X2_warped = unpack_params(pars)
		X_warped = np.vstack([X1_warped, X2_warped])

		data_ax.cla()
		aligned_ax.cla()

		data_ax.scatter(X1_observed[:, 0], X1_observed[:, 1], c=Y1, label="X1")
		data_ax.scatter(X2_observed[:, 0], X2_observed[:, 1], c=Y2, marker="+", label="X2")
		data_ax.legend()

		aligned_ax.scatter(X1_warped[:, 0], X1_warped[:, 1], c=Y1, label="X1")
		aligned_ax.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2, marker="+", label="X2")
		plt.legend()
		plt.draw()
		plt.pause(1.0/60.0)


	# "Parameters" are the warped coordinates
	# param_init = np.random.normal(size=n * 2, scale=0.01)
	param_init = np.ndarray.flatten(X_observed)

	X1_warped, X2_warped = unpack_params(param_init)
	X_warped = np.vstack([X1_warped, X2_warped])

	data_ax.cla()
	aligned_ax.cla()

	data_ax.scatter(X1_observed[:, 0], X1_observed[:, 1], c=Y1, label="X1")
	data_ax.scatter(X2_observed[:, 0], X2_observed[:, 1], c=Y2, marker="+", label="X2")
	data_ax.legend()

	aligned_ax.scatter(X1_warped[:, 0], X1_warped[:, 1], c=Y1, label="X1")
	aligned_ax.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2, marker="+", label="X2")
	plt.legend()
	plt.draw()
	plt.pause(30.0/60.0)

	res = minimize(value_and_grad(gp_likelihood), param_init, jac=True, method='CG', callback=summary)
	pars = res.x

	plt.figure(figsize=(14, 7))

	X1_warped, X2_warped = unpack_params(pars)
	X_warped = np.vstack([X1_warped, X2_warped])

	plt.subplot(121)
	plt.title("Data")
	plt.scatter(X1_observed[:, 0], X1_observed[:, 1], c=Y1)
	plt.scatter(X2_observed[:, 0], X2_observed[:, 1], c=Y2, marker="+")

	plt.subplot(122)
	plt.title("Warped data")
	plt.scatter(X1_warped[:, 0], X1_warped[:, 1], c=Y1)
	plt.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2, marker="+")
	plt.colorbar()
	# plt.savefig("./plots/final_warp.png")
	plt.show()

	#### Predict grid of points
	x1_full, x2_full = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
	X_star = np.vstack([x1_full.ravel(), x2_full.ravel()]).T

	K_nbarnbar = kernel(X_warped, X_warped) + np.eye(n)
	K_starstar = kernel(X_star, X_star)
	K_starnbar = kernel(X_star, X_warped)

	Ystar_mean = K_starnbar @ inv(K_nbarnbar) @ Y

	plt.figure(figsize=(7, 7))
	plt.scatter(X_star[:, 0], X_star[:, 1], c=Ystar_mean, marker="s")
	# plt.savefig("./plots/reference_space_predictions.png")
	plt.show()


	import ipdb; ipdb.set_trace()

