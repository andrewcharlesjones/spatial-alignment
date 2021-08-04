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


class LMC:
    def __init__(
        self,
        X,
        Y,
        kernel,
        n_latent_dims=2,
        n_spatial_dims=2,
        n_kernel_params=2,
        n_noise_variance_params=1,
    ):

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples do not match between X and Y.")

        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.n_latent_dims = n_latent_dims
        self.n_spatial_dims = n_spatial_dims
        self.n_kernel_params = n_kernel_params
        self.n_noise_variance_params = n_noise_variance_params

        self.N = X.shape[0]
        self.n_genes = Y.shape[1]

    def unpack_params(self, params, n_kernel_params):
        noise_variance = np.exp(params[0]) + 0.001
        kernel_params = params[1 : n_kernel_params + 1]
        W = np.reshape(
            params[n_kernel_params + 1:], (self.n_latent_dims, self.n_genes)
        )
        return W, noise_variance, kernel_params

    def gp_likelihood(self, params):
        W, noise_variance, kernel_params = self.unpack_params(
            params, self.n_kernel_params
        )

        # Compute log likelihood
        cov_xx = self.kernel(X, X, kernel_params) + noise_variance * np.eye(self.N)
        cov = np.kron(cov_xx, W.T @ W) + 0.01 * np.eye(self.N * self.n_genes)
        # import ipdb; ipdb.set_trace()
        LL_obs = mvn.logpdf(self.Y.flatten(), np.zeros(self.N * self.n_genes), cov)
        # LL_obs = mvn.logpdf(self.Y, np.zeros((self.N, self.n_genes)), self.kernel(X, X, kernel_params) + noise_variance * np.eye(self.N), W.T @ W)

        return -LL_obs

    def summary(self, pars):
        print("LL {0:1.3e}".format(-self.gp_likelihood(pars)))

        if self.plot_updates:
            W, noise_variance, kernel_params = self.unpack_params(
                pars, self.n_kernel_params
            )

            self.data_ax.cla()
            self.aligned_ax.cla()

            self.data_ax.scatter(
                self.X[:, 0],
                self.X[:, 1],
                c=self.Y[:, 0],
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
        param_init = np.concatenate(
            [
                np.random.normal(size=self.n_noise_variance_params),  # Noise variance
                np.random.normal(size=self.n_kernel_params),  # GP params
                np.random.normal(
                    scale=1, size=self.n_latent_dims * self.n_genes
                ),  # W (loadings)
            ]
        )

        self.plot_updates = plot_updates
        if plot_updates:
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
        W, noise_variance, kernel_params = self.unpack_params(
            res.x, self.n_kernel_params
        )
        return W, noise_variance, kernel_params


if __name__ == "__main__":

    n_genes = 2
    n_latent_dims = 1
    kernel = rbf_covariance
    kernel_params_true = np.array([1, 1.])
    n = 50
    ntest = 50
    sigma2 = .1
    n_spatial_dims = 1
    # X_full = np.hstack([np.random.uniform(low=-5, high=5, size=(n + ntest, 1)) for _ in range(n_spatial_dims)])
    X_full = np.vstack([np.linspace(-7, 7, n + ntest, 1) for _ in range(n_spatial_dims)]).T
    # W_orig = np.random.normal(size=(n_latent_dims, n_genes))
    W_orig =np.array([[-2., 2.]])
    F_orig = np.vstack(
        [
            mvnpy.rvs(mean=np.zeros(n + ntest), cov=kernel(X_full, X_full, kernel_params_true))
            for _ in range(n_latent_dims)
        ]
    ).T
    Y_full = F_orig @ W_orig + np.random.normal(scale=np.sqrt(sigma2), size=(n + ntest, n_genes))

    X = X_full[:n]
    Y = Y_full[:n]
    Xtest = X_full[n:]
    Ytest = Y_full[n:]

    warp_gp = LMC(
        X, Y, kernel=rbf_covariance, n_latent_dims=n_latent_dims, n_spatial_dims=n_spatial_dims
    )
    W_fitted, noise_variance, kernel_params = warp_gp.fit(plot_updates=False)

    ## Make predictions
    nnew = 75
    xnew_lim = 12
    xnew = np.linspace(-xnew_lim, xnew_lim, nnew)
    Xnew = np.expand_dims(xnew, 1)

    WWT = np.outer(W_fitted.T, W_fitted)
    Kxx = np.kron(rbf_covariance(X, X, kernel_params), WWT)
    Kxx += noise_variance * np.eye(Kxx.shape[0])

    Kxnewxnew = np.kron(rbf_covariance(Xnew, Xnew, kernel_params), WWT)
    Kxxnew = np.kron(rbf_covariance(X, Xnew, kernel_params), WWT)
    Kxx_inv = np.linalg.solve(Kxx, np.eye(Kxx.shape[0]))

    # Y_flattened
    mean_pred = Kxxnew.T @ Kxx_inv @ np.ndarray.flatten(Y, "C")
    mean_pred = np.reshape(mean_pred, (nnew, n_genes))


    Xaugmented = np.concatenate([X, Xtest], axis=0)
    Kxx_augmented = rbf_covariance(Xaugmented, Xaugmented, kernel_params)

    Kxx_augmented_full = np.kron(WWT, Kxx_augmented)
    Kxx = Kxx_augmented_full[:n*n_genes + ntest, :n*n_genes + ntest] + 0.01 * np.eye(n * n_genes + ntest)
    
    Kxxtest = Kxx_augmented_full[:n*n_genes + ntest, n*n_genes + ntest:]
    Kxx_inv = np.linalg.solve(Kxx, np.eye(n*n_genes + ntest))
        
    # np.concatenate([np.ndarray.flatten(Y, "F"), Ytest[:, 0]])
    Y_for_preds = np.concatenate([Y[:, 0], Ytest[:, 0], Y[:, 1]])
    preds = Kxxtest.T @ Kxx_inv @ Y_for_preds
    # import ipdb; ipdb.set_trace()
    
    


    ## Get normal GP predictions for the features independently
    Kxx = rbf_covariance(X, X, kernel_params)
    Kxx += noise_variance * np.eye(Kxx.shape[0])
    Kxx_inv = np.linalg.solve(Kxx, np.eye(Kxx.shape[0]))
    Kxxnew = rbf_covariance(X, Xnew, kernel_params)
    mean_pred_gp_y1 = Kxxnew.T @ Kxx_inv @ Y[:, 0]
    mean_pred_gp_y2 = Kxxnew.T @ Kxx_inv @ Y[:, 1]


    plt.figure(figsize=(10, 4))
    # plt.subplot(121)
    # plt.scatter(Y[:, 0], Y[:, 1], c=X[:, 0])
    # plt.xlabel("Y1")
    # plt.ylabel("Y2")
    # plt.colorbar()
    # plt.title("Data")

    # plt.subplot(122)
    plt.scatter(X[:, 0], Y[:, 0], color="red", alpha=0.5, label="Y1")
    # plt.plot(xnew, mean_pred_gp_y1, color="red", alpha=0.5, linestyle='--', label="GP")
    plt.scatter(X[:, 0], Y[:, 1], color="green", alpha=0.5, label="Y2")
    plt.plot(xnew, mean_pred_gp_y2, color="gray", linestyle='--', label="GP predictions")

    # plt.plot(xnew, mean_pred[:, 0], color="red", alpha=0.5, label="LMC")
    # plt.plot(xnew, mean_pred[:, 1], color="green", alpha=0.5, label="LMC")

    plt.plot(Xtest, preds, label="LMC predictions", color="black", linewidth=5)
    plt.scatter(Xtest, Ytest[:, 0], color="red", label="Extra data", alpha=0.5)
    # plt.scatter(Xtest, Ytest[:, 1], color="green", label="Ground truth")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GP regression")

    # plt.subplot(133)
    # plt.scatter(X[:, 0], Y[:, 0], color="red", alpha=0.5)
    # plt.plot(xnew, mean_pred[:, 0], color="red", label="Y1", alpha=0.5)
    # plt.scatter(X[:, 0], Y[:, 1], color="green", alpha=0.5)
    # plt.plot(xnew, mean_pred[:, 1], color="green", label="Y2", alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("LMC")
    plt.legend()
    plt.show()

    import ipdb; ipdb.set_trace()
