import sys

sys.path.append("..")
from warp_gp_pytorch import TwoLayerWarpGP
import numpy as np
from gp_functions import rbf_covariance
from scipy.stats import multivariate_normal as mvnpy
from util import polar_warp


n_views = 2
p = 2
n_genes = 10
kernel = rbf_covariance
kernel_params_true = np.array([1.0, 1.0])
n1, n2 = 30, 30
view_idx = np.array([np.arange(0, n1), np.arange(n1, n1 + n2)])
n = n1 + n2
sigma2 = 1
x1, x2 = np.random.uniform(low=-3, high=3, size=(n1, 1)), np.random.uniform(
    low=-3, high=3, size=(n1, 1)
)
X_orig = np.hstack([x1, x2])
Y_orig = np.vstack(
    [
        mvnpy.rvs(mean=np.zeros(n1), cov=kernel(X_orig, X_orig, kernel_params_true))
        for _ in range(n_genes)
    ]
).T

X1 = X_orig.copy()
Y1 = Y_orig.copy()

X2 = X_orig.copy()
Y2 = Y_orig.copy()

Y = np.concatenate([Y1, Y2], axis=0)

# Warp
linear_coeffs = np.random.normal(scale=0.1, size=n_views * p * 2)
r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

X1_observed = polar_warp(X1, r1s_true, theta1s_true)
X2_observed = polar_warp(X2, r2s_true, theta2s_true)
X = np.vstack([X1_observed, X2_observed])

warp_gp = TwoLayerWarpGP(
    X, Y, n_views=n_views, n_samples_list=[n1, n2], kernel=rbf_covariance
)
warp_gp.fit(plot_updates=True)
