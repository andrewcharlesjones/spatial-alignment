import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys

sys.path.append("../..")
sys.path.append("../../data")
from warps import apply_gp_warp

from gpsa.util import rbf_kernel_numpy as rbf_covariance
from gpsa import polar_warp
from scipy.stats import multivariate_normal as mvnpy


def generate_twod_data(
    n_views,
    n_outputs,
    grid_size,
    n_latent_gps=None,
    kernel_variance=0.1,
    kernel_lengthscale=5,
    noise_variance=0.0,
    fixed_view_idx=None,
):

    kernel = rbf_covariance
    kernel_params_true = [np.log(1.0), np.log(1.0)]
    xlimits = [0, 10]
    ylimits = [0, 10]
    x1s = np.linspace(*xlimits, num=grid_size)
    x2s = np.linspace(*ylimits, num=grid_size)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig_single = np.vstack([X1.ravel(), X2.ravel()]).T
    X_orig = np.concatenate([X_orig_single.copy(), X_orig_single.copy()], axis=0)
    n_samples_per_view = X_orig.shape[0] // 2

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
    K_XX = kernel(X_orig_single, X_orig_single, kernel_params_true)

    nY = n_outputs if n_latent_gps is None else n_latent_gps

    Y_orig = np.vstack(
        [
            mvnpy.rvs(
                mean=np.zeros(X_orig_single.shape[0]),
                cov=K_XX + 0.001 * np.eye(K_XX.shape[0]),
            )
            for _ in range(nY)
        ]
    ).T

    if n_latent_gps is not None:
        W_mat = np.random.normal(size=(n_latent_gps, n_outputs))
        Y_orig = Y_orig @ W_mat

    Y = np.concatenate([Y_orig] * n_views, axis=0)
    X = X_orig.copy()

    # X[n_samples_per_view:] = X[n_samples_per_view:] @ (
    #     np.eye(2) + np.random.normal(0, 0.01, size=(2, 2))
    # )
    # X[:n_samples_per_view] = X[:n_samples_per_view] @ (
    #     np.eye(2) + np.random.normal(0, 0.01, size=(2, 2))
    # )

    X, Y, n_samples_list, view_idx = apply_gp_warp(
        X_orig_single[:n_samples_per_view],
        Y_orig[:n_samples_per_view],
        n_views=2,
        kernel_variance=kernel_variance,
        kernel_lengthscale=kernel_lengthscale,
        noise_variance=noise_variance,
    )
    if fixed_view_idx is not None:
        X[view_idx[fixed_view_idx]] = X_orig_single

    return X, Y, n_samples_list, view_idx


def generate_twod_data_partial_overlap(
    n_views,
    n_outputs,
    grid_size,
    n_latent_gps=None,
    kernel_variance=0.1,
    kernel_lengthscale=5,
    noise_variance=0.0,
):

    kernel = rbf_covariance
    kernel_params_true = [np.log(1.0), np.log(1.0)]
    xlimits = [-5, 5]
    ylimits = [-5, 5]
    x1s = np.linspace(*xlimits, num=grid_size)
    x2s = np.linspace(*ylimits, num=grid_size)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig_single = np.vstack([X1.ravel(), X2.ravel()]).T
    
    ## Only keep the center square of points
    X_orig_single_partial = X_orig_single.copy()
    keep_idx = np.logical_and(np.abs(X_orig_single_partial[:, 0]) < 2.5, np.abs(X_orig_single_partial[:, 1]) < 2.5)
    X_orig_single_partial = X_orig_single_partial[keep_idx]
    X_orig = np.concatenate([X_orig_single.copy(), X_orig_single_partial.copy()], axis=0)
    n_samples_per_view = X_orig.shape[0] // 2

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
    K_XX = kernel(X_orig_single, X_orig_single, kernel_params_true)

    nY = n_outputs if n_latent_gps is None else n_latent_gps

    Y_orig = np.vstack(
        [
            mvnpy.rvs(
                mean=np.zeros(X_orig_single.shape[0]),
                cov=K_XX + 0.001 * np.eye(K_XX.shape[0]),
            )
            for _ in range(nY)
        ]
    ).T

    if n_latent_gps is not None:
        W_mat = np.random.normal(size=(n_latent_gps, n_outputs))
        Y_orig = Y_orig @ W_mat

    # Y = np.concatenate([Y_orig] * n_views, axis=0)
    # Y = np.concatenate(
    #     [
    #         Y_orig,
    #         Y_orig[keep_idx],
    #     ]
    # )
    # X = X_orig.copy()

    # X[n_samples_per_view:] = X[n_samples_per_view:] @ (
    #     np.eye(2) + np.random.normal(0, 0.01, size=(2, 2))
    # )
    # X[:n_samples_per_view] = X[:n_samples_per_view] @ (
    #     np.eye(2) + np.random.normal(0, 0.01, size=(2, 2))
    # )

    X, Y, n_samples_list, view_idx = apply_gp_warp(
        X_orig_single[:grid_size**2],
        Y_orig[:grid_size**2],
        n_views=2,
        kernel_variance=kernel_variance,
        kernel_lengthscale=kernel_lengthscale,
        noise_variance=noise_variance,
    )


    X = np.concatenate(
        [
            X[:grid_size**2],
            X[grid_size**2:][keep_idx],
        ]
    )
    Y = np.concatenate(
        [
            Y[:grid_size**2],
            Y[grid_size**2:][keep_idx],
        ]
    )
    view_idx = view_idx.tolist()
    view_idx[1] = np.where(keep_idx == True)[0]
    n_samples_list[1] = keep_idx.sum()
    # import ipdb; ipdb.set_trace()

    return X, Y, n_samples_list, view_idx, keep_idx
