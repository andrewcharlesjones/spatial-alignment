import numpy as np
import matplotlib.pyplot as plt
inv = np.linalg.inv
import sklearn.gaussian_process.kernels as skernels
from scipy.stats import multivariate_normal as mvn


class GPRegressor:
	def __init__(
		self,
		kernel,
		sigma2: float = 1.0,
		use_inducing_points: bool = False,
		inducing_points=None,
		num_inducing_points=None,
	):

		self.kernel = kernel
		self.use_inducing_points = use_inducing_points
		self.inducing_points = inducing_points
		self.num_inducing_points = num_inducing_points
		self.sigma2 = sigma2

	def fit(self, X, Y):
		self.n, self.p = X.shape
		self.K_XX = self.kernel(X, X)
		self.X = X
		self.Y = Y

		if self.use_inducing_points:

			# Create inducing locations
			if self.num_inducing_points is None:
				self.num_inducing_points = self.n // 5
			Xmin, Xmax = np.min(self.X, axis=0), np.max(self.X, axis=0)
			self.Xbar = np.linspace(Xmin, Xmax, self.num_inducing_points)

			# Form required covariance matrices
			self.K_XbarXbar = self.kernel(self.Xbar, self.Xbar)
			self.K_XbarX = self.kernel(self.Xbar, self.X)

			K_XbarXbar_inv = inv(self.K_XbarXbar)

			self.Lambda_diag = np.diagonal(self.K_XX) - [
				self.K_XbarX[:, ii] @ K_XbarXbar_inv @ self.K_XbarX[:, ii]
				for ii in range(self.n)
			]
			self.Lambda_inv = np.diag(1 / (self.Lambda_diag + self.sigma2))
			self.Q_XbarXbar = (
				self.K_XbarXbar + self.K_XbarX @ self.Lambda_inv @ self.K_XbarX.T
			)

	def predict(self, Xt):

		n_test = Xt.shape[0]
		self.K_XtXt = self.kernel(Xt, Xt)

		if self.use_inducing_points:

			Q_inv = inv(self.Q_XbarXbar)
			# import ipdb; ipdb.set_trace()
			self.K_XtXbar = self.kernel(Xt, self.Xbar)

			# Mean
			mean_firstterm = [
				self.K_XtXbar[ii, :] @ Q_inv for ii in range(n_test)
			] @ self.K_XbarX
			mean_secondterm = self.Lambda_inv @ self.Y
			mean = mean_firstterm @ mean_secondterm

			# Covariance
			middle_term = inv(self.K_XbarXbar) - Q_inv
			middle_term_expanded = np.array(
				[
					self.K_XtXbar[ii, :] @ middle_term @ self.K_XtXbar[ii, :]
					for ii in range(n_test)
				]
			)
			covariance = (
				self.K_XtXt - middle_term_expanded + self.sigma2 * np.eye(n_test)
			)

		else:
			self.K_XtX = self.kernel(Xt, self.X)
			inv_cov = inv(self.K_XX + self.sigma2 * np.eye(self.n))
			mean = self.K_XtX @ inv_cov @ self.Y
			covariance = self.K_XtXt - self.K_XtX @ inv_cov @ self.K_XtX.T

		return mean, covariance

	def plot_predictions(self, Xtest, mean, cov):
		errs = 2 * np.sqrt(np.diag(cov))
		plt.fill_between(Xtest.squeeze(), mean - errs, mean + errs, alpha=0.4)
		plt.plot(Xtest, mean)
		plt.scatter(self.X, self.Y, label="Data")
		plt.legend()
		plt.show()


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import seaborn as sns

	# Toy dataset drawn from GP prior with RBF kernel
	kern = skernels.RBF()
	n = 30
	X = np.random.uniform(low=-3, high=3, size=(n, 1))
	Y = mvn.rvs(mean=np.zeros(n), cov=kern(X, X))

	# GPR
	gpr = GPRegressor(kernel=kern, use_inducing_points=True)
	gpr.fit(X, Y)

	# Plot
	Xtest = np.expand_dims(np.linspace(-3, 3, 200), 1)
	mean_pred, cov_pred = gpr.predict(Xtest)
	gpr.plot_predictions(mean_pred, cov_pred)

	


	import ipdb

	ipdb.set_trace()

	
