# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import sklearn.gaussian_process.kernels as skernels
from scipy.stats import multivariate_normal as mvnpy
from autograd.scipy.stats import multivariate_normal as mvn
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from scipy.spatial import distance_matrix
inv = np.linalg.inv

def polar_warp(X, r, theta):
	return np.array(
		[
			X[:, 0] + r * np.cos(theta),
			X[:, 1] + r * np.sin(theta)
		]).T

def correlation_loss(params, ii):

	r1s, theta1s = X1_observed @ params[:2], X1_observed @ params[2:4]
	r2s, theta2s = X2_observed @ params[4:6], X2_observed @ params[6:]

	X1_warped = polar_warp(X1_observed, r1s, theta1s)
	X2_warped = polar_warp(X2_observed, r2s, theta2s)
	X_warped = np.vstack([X1_warped, X2_warped])

	# Compute log likelihood
	mean = np.zeros(n)

	# Form required covariance matrices
	cov_X1 = kernel(X1_warped, X1_warped)
	cov_X2 = kernel(X2_warped, X2_warped)

	# Conditional means and covs for smoothing
	conditional_mean_X1 = 

	LL = -n / 2.0 * np.log(2. * np.pi) - .5 * np.linalg.slogdet(covariance)[1] - (Y - mean).T @ inv(covariance) @ (Y - mean)

	return -LL

def rbf_covariance(x, xp):
	output_scale = 1 #np.exp(kernel_params[0])
	lengthscales = 1 #np.exp(kernel_params[1:])
	diffs = np.expand_dims(x /lengthscales, 1)\
		  - np.expand_dims(xp/lengthscales, 0)
	return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

if __name__ == "__main__":

	# Toy dataset drawn from GP prior with RBF kernel
	# kernel = skernels.RBF()
	# kernel = lambda X1, X2: X1 @ X2.T + sigma2 #* np.eye(X1.shape[0])
	# kernel = lambda X1, X2: np.exp(-0.5 * np.sum(np.einsum('ij,kj->ikj', X1, -X2)**2, axis=2))

	n_views = 2
	p = 2
	kernel = rbf_covariance
	n1, n2 = 20, 20
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
	linear_coeffs = np.random.normal(scale=1, size=n_views * p * 2)
	# linear_coeffs = np.zeros(n_views * p * 2)
	r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
	r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

	X1_observed = polar_warp(X1, r1s_true, theta1s_true)
	X2_observed = polar_warp(X2, r2s_true, theta2s_true)
	X_observed = np.vstack([X1_observed, X2_observed])

	x1, x2 = np.meshgrid(np.linspace(-10, 10, 5), np.linspace(-10, 10, 5))
	Xbar = np.vstack([x1.ravel(), x2.ravel()]).T
	

	# plt.subplot(121)
	# for ii in range(n//2):
	# 	plt.arrow(x=X1[ii, 0], y=X1[ii, 1], dx=X1_observed[ii, 0] - X1[ii, 0], dy=X1_observed[ii, 1] - X1[ii, 1], head_width=.05)

	# plt.scatter(X1[:, 0], X1[:, 1], label="Original")
	# plt.scatter(X1_observed[:, 0], X1_observed[:, 1], label="Warped")
	# plt.legend()

	# plt.subplot(122)
	# for ii in range(n//2):
	# 	plt.arrow(x=X2[ii, 0], y=X2[ii, 1], dx=X2_observed[ii, 0] - X2[ii, 0], dy=X2_observed[ii, 1] - X2[ii, 1], head_width=.05)

	# plt.scatter(X2[:, 0], X2[:, 1], label="Original")
	# plt.scatter(X2_observed[:, 0], X2_observed[:, 1], label="Warped")
	# plt.legend()
	# plt.show()

	distance_mat_true = distance_matrix(X_orig, X_orig)

	# param_list = np.concatenate([rs, thetas])
	# param_list = np.concatenate([r1s, theta1s, r2s, theta2s])

	# offset_list = np.linspace(-10, 10, 100)
	# liks = [gp_likelihood(np.concatenate([np.ones(n) * xx, np.zeros(n)]), 0) for xx in offset_list]
	# plt.plot(offset_list, liks)
	# plt.show()

	# rs_fitted, thetas_fitted = param_list[:n], param_list[n:]
	# X_warped = polar_warp(X, rs_fitted, thetas_fitted)
	# plt.scatter(X_warped[:n//2, 0], X_warped[:n//2, 1], c=Y[:n//2])
	# plt.scatter(X_warped[n//2:, 0], X_warped[n//2:, 1], c=Y[n//2:], marker="^")
	# plt.show()

	LL_grad = grad(gp_likelihood)

	fig = plt.figure(figsize=(14, 7), facecolor='white')
	data_ax = fig.add_subplot(121, frameon=False)
	aligned_ax = fig.add_subplot(122, frameon=False)
	plt.show(block=False)

	def summary(pars, step, gradient):
		print('step {0:5d}: {1:1.3e}'.format(step, gp_likelihood(pars, step)))
		# rs_fitted, thetas_fitted = pars[:n], pars[n:]
		# rs_fitted, thetas_fitted = X @ pars[:2], X @ pars[2:]
		# X_warped = polar_warp(X, rs_fitted, thetas_fitted)

		r1s, theta1s = X1 @ pars[:2], X1 @ pars[2:4]
		r2s, theta2s = X2 @ pars[4:6], X2 @ pars[6:]

		X1_warped = polar_warp(X1, r1s, theta1s)
		X2_warped = polar_warp(X2, r2s, theta2s)
		# X2_warped = X2_observed
		X_warped = np.vstack([X1_warped, X2_warped])

		# plt.figure(figsize=(14, 7))
		data_ax.cla()
		aligned_ax.cla()

		# plt.subplot(121)
		data_ax.scatter(X1[:, 0], X1[:, 1], c=Y1)

		# plt.subplot(122)
		aligned_ax.scatter(X1_warped[:, 0], X1_warped[:, 1], c=Y1, label="X1")
		aligned_ax.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2, marker="^", label="X2")
		plt.legend()
		plt.draw()
		plt.pause(1.0/60.0)

		distance_mat_X1 = distance_matrix(X1_warped, X1_warped)
		distance_mat_X2 = distance_matrix(X2_warped, X2_warped)
		loss = np.mean((distance_mat_X1 - distance_mat_true)**2)
		print("{0:1.4}".format(loss))

		# import ipdb; ipdb.set_trace()


	param_init = np.random.normal(size=n_views * p * 2, scale=0.01)

	# plt.figure(figsize=(14, 7))

	# r1s, theta1s = X1 @ param_init[:2], X1 @ param_init[2:4]
	# r2s, theta2s = X2 @ param_init[4:6], X2 @ param_init[6:]

	# X1_warped = polar_warp(X1_observed, r1s, theta1s)
	# X2_warped = polar_warp(X2_observed, r2s, theta2s)
	# X_warped = np.vstack([X1_warped, X2_warped])

	# plt.subplot(121)
	# plt.scatter(X1[:, 0], X1[:, 1], c=Y1)

	# plt.subplot(122)
	# plt.scatter(X1_warped[:, 0], X1_warped[:, 1], c=Y1)
	# plt.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2, marker="^")
	# plt.show()


	pars = adam(LL_grad, param_init, step_size=0.1, num_iters=200, callback=summary)

	plt.figure(figsize=(14, 7))

	r1s, theta1s = X1 @ pars[:2], X1 @ pars[2:4]
	r2s, theta2s = X2 @ pars[4:6], X2 @ pars[6:]

	X1_warped = polar_warp(X1_observed, r1s, theta1s)
	X2_warped = polar_warp(X2_observed, r2s, theta2s)
	X_warped = np.vstack([X1_warped, X2_warped])

	# plt.subplot(121)
	# plt.scatter(X1[:, 0], X1[:, 1], c=Y1)

	# plt.subplot(122)
	# plt.scatter(X1_warped[:, 0], X1_warped[:, 1], c=Y1)
	# # plt.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2, marker="^")
	# plt.scatter(X2_observed[:, 0], X2_observed[:, 1], c=Y2, marker="^")
	# plt.show()

	plt.subplot(121)
	plt.scatter(X1[:, 0], X1[:, 1], c=np.arange(n//2))

	plt.subplot(122)
	plt.scatter(X1_warped[:, 0], X1_warped[:, 1], c=np.arange(n//2))
	# plt.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2, marker="^")
	plt.scatter(X2_observed[:, 0], X2_observed[:, 1], c=np.arange(n//2), marker="^")
	plt.colorbar()
	plt.show()

	# plt.scatter(linear_coeffs, pars)
	# plt.xlabel("True parameters")
	# plt.ylabel("Estimated parameters")
	# plt.show()

	import ipdb; ipdb.set_trace()

