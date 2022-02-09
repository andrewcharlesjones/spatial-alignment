import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from gpsa.util import rbf_kernel_numpy as rbf_covariance
from scipy.stats import multivariate_normal as mvnpy


def generate_oned_data_affine_warp(
    n_views,
    n_outputs,
    n_samples_per_view,
    noise_variance=0.0,
    n_latent_gps=None,
    scale_factor=1.1,
    additive_factor=0.3,
):

    kernel = rbf_covariance
    kernel_params_true = np.array([np.log(1.0), np.log(1.0)])
    n_latent_gps = 2
    n_spatial_dims = 1

    X_orig_single = np.random.uniform(-10, 10, size=(n_samples_per_view, 1))
    # X_orig_single = np.linspace(-10, 10, n_samples_per_view)[:, :-1]
    X_orig = np.concatenate([X_orig_single.copy(), X_orig_single.copy()], axis=0)

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

    nY = n_outputs if n_latent_gps is None else n_latent_gps

    Y_orig = np.vstack(
        [
            mvnpy.rvs(
                mean=np.zeros(X_orig_single.shape[0]),
                cov=kernel(X_orig_single, X_orig_single, kernel_params_true),
            )
            for _ in range(nY)
        ]
    ).T

    if n_latent_gps is not None:
        W_mat = np.random.normal(size=(n_latent_gps, n_outputs))
        # W_mat = np.expand_dims(np.array([1, -1]), 0)
        Y_orig = Y_orig @ W_mat

    Y = np.concatenate([Y_orig, Y_orig], axis=0)
    Y += np.random.normal(scale=np.sqrt(noise_variance), size=(Y.shape))
    X = X_orig.copy()
    X[n_samples_per_view:] = X[n_samples_per_view:] * scale_factor + additive_factor

    return X, Y, n_samples_list, view_idx


def generate_oned_data_gp_warp(
    n_views,
    n_outputs,
    n_samples_per_view,
    noise_variance=0.0,
    n_latent_gps=None,
    kernel_variance=1.0,
    kernel_lengthscale=1.0,
    mean_slope=1.,
    mean_intercept=0.,
):

    kernel = rbf_covariance
    kernel_params_true = np.array([np.log(1.0), np.log(1.0)])
    n_spatial_dims = 1

    # X_orig_single = np.random.uniform(-10, 10, size=(n_samples_per_view, 1))
    X_orig_single = np.linspace(-10, 10, n_samples_per_view).reshape(-1, 1)
    X_orig = np.concatenate([X_orig_single.copy()] * n_views, axis=0)

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

    nY = n_outputs if n_latent_gps is None else n_latent_gps

    Y_orig = np.vstack(
        [
            mvnpy.rvs(
                mean=np.zeros(X_orig_single.shape[0]),
                cov=kernel(X_orig_single, X_orig_single, kernel_params_true),
            )
            for _ in range(nY)
        ]
    ).T

    if n_latent_gps is not None:
        if n_outputs == 2:
            W_mat = np.expand_dims(np.array([1, -1]), 0)
        else:
            W_mat = np.random.normal(size=(n_latent_gps, n_outputs))
        
        Y_orig = Y_orig @ W_mat

    Y = np.concatenate([Y_orig] * n_views, axis=0)
    Y += np.random.normal(scale=np.sqrt(noise_variance), size=(Y.shape))
    X = X_orig.copy()

    # X_view1 = X[:n_samples_per_view]
    # X_view2 = X[n_samples_per_view : n_samples_per_view * 2]

    # Draw warped coordinates from a GP
    warp_kernel_params_true = np.array([np.log(kernel_variance), np.log(kernel_lengthscale)])

    for vv in range(n_views):
        X_curr_view_warped = mvnpy.rvs(
            mean=X_orig_single.squeeze() * mean_slope + mean_intercept,
            cov=kernel(X_orig_single, X_orig_single, warp_kernel_params_true),
        )
        X_curr_view_warped = np.expand_dims(X_curr_view_warped, 1)
        # import ipdb; ipdb.set_trace()
        X[n_samples_per_view * vv : n_samples_per_view * (vv + 1)] = X_curr_view_warped

    # X += np.random.normal(scale=np.sqrt(0.1), size=(X.shape))
    # X_view1_warped = mvnpy.rvs(
    #     mean=X_view2.squeeze() * mean_slope + mean_intercept,
    #     cov=kernel(X_view1, X_view1, warp_kernel_params_true),
    # )
    # X_view1_warped = np.expand_dims(X_view1_warped, 1)
    # X[n_samples_per_view : n_samples_per_view * 2] = X_view1_warped

    # X_view2_warped = mvnpy.rvs(
    #     mean=X_view2.squeeze() * mean_slope + mean_intercept,
    #     cov=kernel(X_view2, X_view2, warp_kernel_params_true),
    # )
    # X_view2_warped = np.expand_dims(X_view2_warped, 1)
    # X[n_samples_per_view : n_samples_per_view * 2] = X_view2_warped

    return X, Y, n_samples_list, view_idx


if __name__ == "__main__":

    n_views = 2
    n_samples_per_view = 100
    X, Y, n_samples_list, view_idx = generate_oned_data_gp_warp(
        n_views=n_views, n_outputs=1, n_samples_per_view=n_samples_per_view
    )

    for vv in range(n_views):
        curr_start_idx = vv * n_samples_per_view
        curr_end_idx = vv * n_samples_per_view + n_samples_per_view
        plt.scatter(X[curr_start_idx:curr_end_idx], Y[curr_start_idx:curr_end_idx])
    plt.show()

    import ipdb

    ipdb.set_trace()
