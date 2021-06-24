import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy
from autograd.scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_normal as mvno
import autograd.numpy as np
from autograd import grad, value_and_grad
from autograd.misc.optimizers import adam
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

def linear_warp(X, shifts1, shifts2):
	return np.array(
		[
			X[:, 0] + shifts1,
			X[:, 1] + shifts2
		]).T

def polar_warp_ag(X, param_matrix):
	return X + param_matrix


def distance_matrix(X, Y):
	return np.expand_dims(X, 0)**2 - np.expand_dims(Y, 1)**2

# def gp_likelihood(params, ii):

# 	# rs, thetas = np.exp(params[:n]), np.exp(params[n:])
# 	# rs, thetas = params[:n], params[n:]
# 	# rs, thetas = X @ params[:2], X @ params[2:]

# 	# X_warped = polar_warp(X, rs, thetas)

# 	r1s, theta1s = X1_observed @ params[:2], X1_observed @ params[2:4]
# 	r2s, theta2s = X2_observed @ params[4:6], X2_observed @ params[6:]

# 	X1_warped = polar_warp(X1_observed, r1s, theta1s)
# 	# X2_warped = polar_warp(X2_observed, r2s, theta2s)
# 	X2_warped = X2_observed
# 	X_warped = np.vstack([X1_warped, X2_warped])


# 	# Compute expected log likelihood

# 	# Form required covariance matrices
# 	K_XX = kernel(X_warped, X_warped)
# 	K_XbarXbar = kernel(Xbar, Xbar)
# 	K_XbarX = kernel(Xbar, X_warped)
# 	K_XbarXbar_inv = inv(K_XbarXbar)

# 	# Lambda_diag = np.diagonal(K_XX) - [
# 	# 	K_XbarX[:, ii] @ K_XbarXbar_inv @ K_XbarX[:, ii]
# 	# 	for ii in range(n)
# 	# ]
# 	Lambda_diag = np.diag(K_XX) - np.diag(K_XbarX.T @ K_XbarXbar_inv @ K_XbarX)
# 	Lambda_inv = np.diag(1 / (Lambda_diag + sigma2))
# 	Q_XbarXbar = (
# 		K_XbarXbar + K_XbarX @ Lambda_inv @ K_XbarX.T
# 	)

# 	# Compute mean and covariance
# 	Q_inv = inv(Q_XbarXbar)

# 	# Mean
# 	# mean_firstterm = [
# 	# 	K_XbarX[:, ii] @ Q_inv for ii in range(n)
# 	# ] @ K_XbarX
# 	mean_firstterm = K_XbarX.T @ Q_inv @ K_XbarX
# 	mean_secondterm = Lambda_inv @ Y
# 	mean = mean_firstterm @ mean_secondterm

# 	# Covariance
# 	middle_term = inv(K_XbarXbar) - Q_inv
# 	middle_term_expanded = K_XbarX.T @ middle_term @ K_XbarX
# 	covariance = (
# 		K_XX - middle_term_expanded + sigma2 * np.eye(n)
# 	)

# 	# import ipdb; ipdb.set_trace()
# 	# LL = mvn.logpdf(mean=mean, cov=covariance, x=Y)
# 	LL = -n / 2.0 * np.log(2. * np.pi) - .5 * np.linalg.slogdet(covariance)[1] - (Y - mean).T @ inv(covariance) @ (Y - mean)

# 	return -LL

def gp_likelihood(params):

	# rs, thetas = np.exp(params[:n]), np.exp(params[n:])
	# rs, thetas = params[:n], params[n:]
	# rs, thetas = X @ params[:2], X @ params[2:]

	# X_warped = polar_warp(X, rs, thetas)

	# x11s, x12s = X1_observed[:, 0] * params[0], X1_observed[:, 1] * params[1]
	# x21s, x22s = X2_observed[:, 0] * params[2], X2_observed[:, 1] * params[3]

	r1s, theta1s = np.matmul(X1_observed, params[:2]), np.matmul(X1_observed, params[2:4])
	r2s, theta2s = np.matmul(X2_observed, params[4:6]), np.matmul(X2_observed, params[6:])

	X1_warped = polar_warp(X1_observed, r1s, theta1s)
	X2_warped = polar_warp(X2_observed, r2s, theta2s)
	# X1_warped = linear_warp(X1_observed, x11s, x12s)
	# X2_warped = linear_warp(X2_observed, x21s, x22s)

	# import ipdb; ipdb.set_trace()
	X_warped = np.vstack([X1_warped, X2_warped])

	# Compute log likelihood
	mean = np.zeros(n)

	# Form required covariance matrices
	covariance = kernel(X_warped, X_warped) + np.eye(n)

	# LL = -0.5 * (n * np.log(2. * np.pi) + np.linalg.slogdet(covariance)[1] + (Y - mean).T @ inv(covariance) @ (Y - mean))
	LL = mvn.logpdf(Y, mean, covariance)

	# Penalty for preserving pairwise distances between points
	distance_mat_X1_warped = distance_matrix(X1_warped, X1_warped)
	distance_mat_X2_warped = distance_matrix(X2_warped, X2_warped)
	X1_penalty = np.sum((distance_mat_X1_warped - distance_mat_X1)**2)
	X2_penalty = np.sum((distance_mat_X2_warped - distance_mat_X2)**2)
	distance_penalty = X1_penalty + X2_penalty

	# print(-LL, distance_penalty)
	# import ipdb; ipdb.set_trace()

	return -LL + 1e-5 * distance_penalty

def rbf_covariance(x, xp):
	output_scale = 1 #np.exp(kernel_params[0])
	lengthscales = 1 #np.exp(kernel_params[1:])
	diffs = np.expand_dims(x / lengthscales, 1)\
		  - np.expand_dims(xp / lengthscales, 0)
	return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

if __name__ == "__main__":

	n_views = 2
	p = 2
	kernel = rbf_covariance
	n1, n2 = 200, 200
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
	linear_coeffs = np.random.normal(scale=.1, size=n_views * p * 2)
	# linear_coeffs = np.zeros(n_views * p * 2)
	r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
	r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

	X1_observed = polar_warp(X1, r1s_true, theta1s_true)
	X2_observed = polar_warp(X2, r2s_true, theta2s_true)
	X_observed = np.vstack([X1_observed, X2_observed])

	x1, x2 = np.meshgrid(np.linspace(-10, 10, 5), np.linspace(-10, 10, 5))
	Xbar = np.vstack([x1.ravel(), x2.ravel()]).T



	# x11s, x12s = X1_observed[:, 0] * linear_coeffs[0], X1_observed[:, 1] * linear_coeffs[1]
	# x21s, x22s = X2_observed[:, 0] * linear_coeffs[2], X2_observed[:, 1] * linear_coeffs[3]

	# X1_warped = linear_warp(X1, x11s, x12s)
	# X2_warped = linear_warp(X2, x21s, x22s)
	# plt.scatter(X1[:, 0], X1[:, 1], label="Reference")
	# plt.scatter(X1_warped[:, 0], X1_warped[:, 1], label="Warped")
	# plt.legend()
	# plt.show()

	# import ipdb; ipdb.set_trace()
	

	# plt.figure(figsize=(21, 7))
	# plt.subplot(131)
	# plt.title("Original")
	# plt.xlabel("x1")
	# plt.ylabel("x2")
	# plt.scatter(X1[:, 0], X1[:, 1])

	# plt.subplot(132)
	# plt.title("View 1")
	# plt.xlabel("x1")
	# plt.ylabel("x2")
	# for ii in range(n//2):
	# 	plt.arrow(x=X1[ii, 0], y=X1[ii, 1], dx=X1_observed[ii, 0] - X1[ii, 0], dy=X1_observed[ii, 1] - X1[ii, 1], head_width=.05)

	# plt.scatter(X1[:, 0], X1[:, 1], label="Original")
	# plt.scatter(X1_observed[:, 0], X1_observed[:, 1], label="Warped")
	# plt.legend()

	# plt.subplot(133)
	# plt.title("View 2")
	# plt.xlabel("x1")
	# plt.ylabel("x2")
	# for ii in range(n//2):
	# 	plt.arrow(x=X2[ii, 0], y=X2[ii, 1], dx=X2_observed[ii, 0] - X2[ii, 0], dy=X2_observed[ii, 1] - X2[ii, 1], head_width=.05)

	# plt.scatter(X2[:, 0], X2[:, 1], label="Original")
	# plt.scatter(X2_observed[:, 0], X2_observed[:, 1], label="Warped")
	# plt.legend()
	# plt.savefig("./plots/warped_arrows.png")
	# plt.show()

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

		r1s, theta1s = X1_observed @ pars[:2], X1_observed @ pars[2:4]
		r2s, theta2s = X2_observed @ pars[4:6], X2_observed @ pars[6:]

		X1_warped = polar_warp(X1_observed, r1s, theta1s)
		X2_warped = polar_warp(X2_observed, r2s, theta2s)
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


	param_init = np.random.normal(size=n_views * p * 2, scale=0.01)

	r1s, theta1s = X1_observed @ param_init[:2], X1_observed @ param_init[2:4]
	r2s, theta2s = X2_observed @ param_init[4:6], X2_observed @ param_init[6:]

	X1_warped = polar_warp(X1_observed, r1s, theta1s)
	X2_warped = polar_warp(X2_observed, r2s, theta2s)
	X_warped = np.vstack([X1_warped, X2_warped])

	data_ax.cla()
	aligned_ax.cla()

	data_ax.scatter(X1[:, 0], X1[:, 1], c=Y1)

	aligned_ax.scatter(X1_warped[:, 0], X1_warped[:, 1], c=Y1, label="X1")
	aligned_ax.scatter(X2_warped[:, 0], X2_warped[:, 1], c=Y2, marker="^", label="X2")
	plt.legend()
	plt.draw()
	plt.pause(30.0/60.0)

	res = minimize(value_and_grad(gp_likelihood), param_init, jac=True, method='CG', callback=summary)
	pars = res.x

	plt.figure(figsize=(14, 7))

	r1s, theta1s = X1_observed @ pars[:2], X1_observed @ pars[2:4]
	r2s, theta2s = X2_observed @ pars[4:6], X2_observed @ pars[6:]

	X1_warped = polar_warp(X1_observed, r1s, theta1s)
	X2_warped = polar_warp(X2_observed, r2s, theta2s)
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
	plt.savefig("./plots/final_warp.png")
	plt.show()



	#### Predict grid of points
	x1_full, x2_full = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
	X_star = np.vstack([x1_full.ravel(), x2_full.ravel()]).T

	K_nbarnbar = kernel(X_warped, X_warped) + np.eye(n)
	K_starstar = kernel(X_star, X_star)
	K_starnbar = kernel(X_star, X_warped)

	Ystar_mean = K_starnbar @ inv(K_nbarnbar) @ Y

	plt.figure(figsize=(7, 7))
	plt.scatter(X_star[:, 0], X_star[:, 1], c=Ystar_mean)
	plt.savefig("./plots/reference_space_predictions.png")
	plt.show()


	import ipdb; ipdb.set_trace()

