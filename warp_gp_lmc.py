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
from util import polar_warp
from gp_functions import rbf_covariance, matrix_normal_logpdf
from scipy.optimize import minimize
from sklearn.decomposition import PCA

inv = np.linalg.inv

import matplotlib

font = {"size": 10}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True


class TwoLayerWarpGPLMC:
    def __init__(
        self,
        X,
        Y,
        n_views,
        n_samples_list,
        kernel,
        n_latent_dims=2,
        n_spatial_dims=2,
        n_kernel_params=4,
        n_noise_variance_params=1,
    ):

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples do not match between X and Y.")

        self.X = X
        self.Y = Y
        self.n_views = n_views
        self.n_samples_list = n_samples_list
        self.kernel = kernel
        self.n_latent_dims = n_latent_dims
        self.n_spatial_dims = n_spatial_dims
        self.n_kernel_params = n_kernel_params
        self.n_noise_variance_params = n_noise_variance_params

        self.N = np.sum(n_samples_list)
        self.n_genes = Y.shape[1]
        cumulative_sums = np.cumsum(n_samples_list)
        cumulative_sums = np.insert(cumulative_sums, 0, 0)
        self.view_idx = np.array(
            [
                np.arange(cumulative_sums[ii], cumulative_sums[ii + 1])
                for ii in range(self.n_views)
            ]
        )

    def unpack_params(self, params, n_kernel_params):
        noise_variance = np.exp(params[0]) + 0.001
        kernel_params = params[1 : n_kernel_params + 1]
        mean_params = np.reshape(
            params[
                n_kernel_params + 1 : n_kernel_params + 1 + self.n_spatial_dims ** 2
            ],
            (self.n_spatial_dims, self.n_spatial_dims),
        )

        latent_vars = params[n_kernel_params + self.n_spatial_dims ** 2 + 1 :]
        # n_F = self.N * self.n_latent_dims
        n_W = self.n_latent_dims * self.n_genes
        # F = np.reshape(
        #     latent_vars[:n_F], (self.N, self.n_latent_dims)
        # )
        W = np.reshape(
            latent_vars[:n_W], (self.n_latent_dims, self.n_genes)
        )
        X_warped = np.reshape(
            latent_vars[n_W:], (self.N, self.n_spatial_dims)
        )
        return X_warped, W, noise_variance, kernel_params, mean_params

    def gp_likelihood(self, params):
        X_warped, W, noise_variance, kernel_params, mean_params = self.unpack_params(
            params, self.n_kernel_params
        )

        kernel_params_warp_model = kernel_params[: self.n_kernel_params // 2]
        kernel_params_obs_model = kernel_params[self.n_kernel_params // 2 :]

        # Compute log likelihood
        mean_warp = self.X @ mean_params
        mean_lmc = np.zeros(self.N)
        mean_obs = np.zeros((self.N, self.n_genes))

        # Form required covariance matrices
        # LMC
        covariance_obs = np.kron(self.kernel(X_warped, X_warped, kernel_params_obs_model), W.T @ W)
        covariance_obs += noise_variance * np.eye(self.N * self.n_genes)

        # Warp log likelihood
        LL_warp = 0
        for vv in range(self.n_views):
            curr_view_idx = self.view_idx[vv]
            curr_n = curr_view_idx.shape[0]
            curr_cov_data = self.kernel(
                self.X[curr_view_idx, :],
                self.X[curr_view_idx, :],
                kernel_params_warp_model,
            )
            curr_covariance_warp = curr_cov_data + 0.01 * np.eye(curr_n)
            LL_warp += np.sum(
                [
                    mvn.logpdf(
                        X_warped[curr_view_idx, dd],
                        mean_warp[curr_view_idx, dd],
                        curr_covariance_warp,
                    )
                    for dd in range(self.n_spatial_dims)
                ]
            )
        
        LL_obs = matrix_normal_logpdf(self.Y, np.zeros((self.N, self.n_genes)), self.kernel(X_warped, X_warped, kernel_params_obs_model) + 0.01 * np.eye(self.N), W.T @ W)

        return -LL_warp - LL_obs

    def summary(self, pars):
        print("LL {0:1.3e}".format(-self.gp_likelihood(pars)))

        if self.plot_updates:
            X_warped, W, noise_variance, kernel_params, mean_params = self.unpack_params(
                pars, self.n_kernel_params
            )

            self.data_ax.cla()
            self.aligned_ax.cla()
            markers = [".", "+", "^"]

            for vv in range(self.n_to_plot):
                self.data_ax.scatter(
                    self.X[self.view_idx[vv], 0],
                    self.X[self.view_idx[vv], 1],
                    c=self.Y[self.view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                )
                self.aligned_ax.scatter(
                    X_warped[self.view_idx[vv], 0],
                    X_warped[self.view_idx[vv], 1],
                    c=self.Y[self.view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                )

            self.data_ax.legend(loc="upper left")
            self.aligned_ax.legend(loc="upper left")
            self.data_ax.set_xlabel("Spatial dim 1")
            self.data_ax.set_ylabel("Spatial dim 2")
            self.aligned_ax.set_xlabel("Spatial dim 1")
            self.aligned_ax.set_ylabel("Spatial dim 2")
            plt.draw()
            plt.pause(1.0 / 60.0)

    def fit(self, plot_updates=False):
        Y_pca = PCA(n_components=self.n_latent_dims).fit(self.Y)
        # import ipdb; ipdb.set_trace()
        param_init = np.concatenate(
            [
                np.random.normal(size=self.n_noise_variance_params),  # Noise variance
                np.random.normal(size=self.n_kernel_params),  # GP params
                np.random.normal(size=self.n_spatial_dims ** 2),  # Mean parameters
                # np.random.normal(
                #     scale=1, size=self.N * self.n_latent_dims
                # ),  # F (latent function evaluations) # TODO: Initialize this with PCA
                # Y_pca.transform(self.Y).flatten(),
                np.random.normal(
                    scale=1, size=self.n_latent_dims * self.n_genes
                ),  # W (loadings)
                # Y_pca.components_.flatten(),
                np.ndarray.flatten(self.X)
                + np.random.normal(
                    scale=0.001, size=self.N * self.n_spatial_dims
                ),  # Warped coordinates
            ]
        )

        self.plot_updates = plot_updates
        if plot_updates:
            self.n_to_plot = min(self.n_views, 3)
            fig = plt.figure(figsize=(14, 7), facecolor="white")
            self.data_ax = fig.add_subplot(121, frameon=False)
            self.aligned_ax = fig.add_subplot(122, frameon=False)
            plt.show(block=False)

            # Plot alignment based on initial params
            self.summary(param_init)
        res = minimize(
            value_and_grad(self.gp_likelihood),
            param_init,
            jac=True,
            method="CG",
            callback=self.summary,
        )
        X_warped, W, noise_variance, kernel_params, mean_params = self.unpack_params(
            res.x, self.n_kernel_params
        )


if __name__ == "__main__":

    n_views = 2
    n_genes = 20
    n_latent_dims = 2
    kernel = rbf_covariance
    kernel_params_true = np.array([1., 1.])
    n_samples_per_view = 20
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
    X_orig = np.hstack([np.random.uniform(low=-3, high=3, size=(n_samples_per_view, 1)) for _ in range(2)])
    W_orig = np.random.normal(size=(n_latent_dims, n_genes))
    F_orig = np.vstack(
        [
            mvnpy.rvs(mean=np.zeros(n_samples_per_view), cov=kernel(X_orig, X_orig, kernel_params_true))
            for _ in range(n_latent_dims)
        ]
    ).T
    Y_orig = F_orig @ W_orig + np.random.normal(size=(n_samples_per_view, n_genes))


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
	    Y[view_idx[vv]] = curr_Y

    warp_gp = TwoLayerWarpGPLMC(
        X, Y, n_views=n_views, n_samples_list=n_samples_list, kernel=rbf_covariance, n_latent_dims=n_latent_dims
    )
    warp_gp.fit(plot_updates=True)

    import ipdb; ipdb.set_trace()
