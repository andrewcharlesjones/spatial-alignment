import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys

sys.path.append("../..")


# from gp_functions import rbf_covariance
from gpsa.util import rbf_kernel_numpy as rbf_covariance
from gpsa import polar_warp
from scipy.stats import multivariate_normal as mvnpy


def apply_gp_warp(
    X_orig_single,
    Y_orig_single,
    n_views,
    # n_samples_per_view,
    noise_variance=0.0,
    kernel_variance=1.0,
    kernel_lengthscale=1.0,
    mean_slope=1.0,
    mean_intercept=0.0,
):

    n_samples_per_view = X_orig_single.shape[0]
    n_spatial_dims = X_orig_single.shape[1]

    kernel = rbf_covariance
    kernel_params_true = np.array([np.log(1.0), np.log(1.0)])

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

    X = X_orig.copy()

    # Draw warped coordinates from a GP
    warp_kernel_params_true = np.array(
        [np.log(kernel_variance), np.log(kernel_lengthscale)]
    )

    for vv in range(n_views):

        for ss in range(n_spatial_dims):
            X_curr_view_warped = mvnpy.rvs(
                mean=X_orig_single[:, ss] * mean_slope + mean_intercept,
                cov=kernel(X_orig_single, X_orig_single, warp_kernel_params_true),
            )
            # import ipdb; ipdb.set_trace()
            X[
                n_samples_per_view * vv : n_samples_per_view * (vv + 1), ss
            ] = X_curr_view_warped

    Y = np.concatenate([Y_orig_single] * n_views, axis=0)
    Y += np.random.normal(scale=np.sqrt(noise_variance), size=Y.shape)

    return X, Y, n_samples_list, view_idx


def apply_gp_warp_multimodal(
    X_orig_singles,
    Y_orig_singles,
    n_views,
    # n_samples_per_view,
    noise_variance=0.0,
    kernel_variance=1.0,
    kernel_lengthscale=1.0,
    mean_slope=1.0,
    mean_intercept=0.0,
):
    assert len(X_orig_singles) == len(Y_orig_singles)

    n_modalities = len(X_orig_singles)

    modality_idx = np.cumsum([x.shape[0] for x in X_orig_singles])
    modality_idx = np.insert(modality_idx, 0, 0)

    kernel = rbf_covariance
    kernel_params_true = np.array([np.log(1.0), np.log(1.0)])

    X_orig_single = np.concatenate(X_orig_singles, axis=0)

    X_orig_single = X_orig_single - X_orig_single.min(0)
    X_orig_single = X_orig_single / X_orig_single.max(0)
    X_orig_single *= 10

    X_orig = np.concatenate([X_orig_single.copy()] * n_views, axis=0)

    n_samples_per_view = X_orig_single.shape[0]
    n_spatial_dims = X_orig_single.shape[1]

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

    X = X_orig.copy()

    # Draw warped coordinates from a GP
    warp_kernel_params_true = np.array(
        [np.log(kernel_variance), np.log(kernel_lengthscale)]
    )

    for vv in range(n_views):

        curr_idx = np.arange(n_samples_per_view * vv, n_samples_per_view * (vv + 1))

        for ss in range(n_spatial_dims):
            X_curr_view_warped = mvnpy.rvs(
                mean=X_orig_single[:, ss] * mean_slope + mean_intercept,
                cov=kernel(X_orig_single, X_orig_single, warp_kernel_params_true),
            )
            # import ipdb; ipdb.set_trace()
            X[curr_idx, ss] = X_curr_view_warped

    view_idx = np.cumsum([n_samples_per_view * vv for vv in range(n_views + 1)])

    X_warped = []
    Y_warped = []
    n_samples_list = []
    for mm in range(n_modalities):

        curr_modality_idx = np.concatenate(
            [
                view_idx[vv] + np.arange(modality_idx[mm], modality_idx[mm + 1])
                for vv in range(n_views)
            ]
        )
        X_warped.append(X[curr_modality_idx])

        Y_full_mm = np.concatenate([Y_orig_singles[mm]] * n_views, axis=0)
        Y_full_mm += np.random.normal(
            scale=np.sqrt(noise_variance), size=Y_full_mm.shape
        )
        Y_warped.append(Y_full_mm)
        n_samples_list.append([X_orig_singles[mm].shape[0]] * n_views)

    return X_warped, Y_warped, n_samples_list, view_idx


def apply_linear_warp(
    X_orig_single,
    Y_orig_single,
    n_views,
    linear_slope_variance=0.1,
    linear_intercept_variance=0.1,
    noise_variance=0.01,
    rotation=True,
):

    n_samples_per_view = X_orig_single.shape[0]
    n_spatial_dims = X_orig_single.shape[1]

    kernel = rbf_covariance
    kernel_params_true = np.array([np.log(1.0), np.log(1.0)])

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

    X = X_orig.copy()

    for vv in range(n_views):

        # curr_slopes = np.eye(n_spatial_dims) + np.random.normal(
        #     loc=0,
        #     scale=np.sqrt(linear_slope_variance),
        #     size=(n_spatial_dims, n_spatial_dims),
        # )
        # import ipdb; ipdb.set_trace()
        # curr_slopes /= np.linalg.norm(curr_slopes, ord=2, axis=0)
        # curr_slopes = np.linalg.svd(curr_slopes)[0]

        # curr_slopes = np.random.normal(
        #     loc=1,
        #     scale=np.sqrt(linear_slope_variance),
        #     size=n_spatial_dims,
        # )

        # curr_intercepts = np.random.normal(
        #     loc=0, scale=np.sqrt(linear_intercept_variance), size=n_spatial_dims
        # )

        curr_slopes = np.random.uniform(
            low=1 - linear_slope_variance,
            high=1 + linear_slope_variance,
            size=n_spatial_dims,
        )

        curr_intercepts = np.random.uniform(
            low=linear_intercept_variance,
            high=linear_intercept_variance,
            size=n_spatial_dims,
        )
        # print(curr_slopes, curr_intercepts)

        X_curr_view_warped = X_orig_single * curr_slopes + curr_intercepts
        X[
            n_samples_per_view * vv : n_samples_per_view * (vv + 1), :
        ] = X_curr_view_warped

    Y = np.concatenate([Y_orig_single] * n_views, axis=0)
    Y += np.random.normal(scale=np.sqrt(noise_variance), size=Y.shape)

    return X, Y, n_samples_list, view_idx


def apply_polar_warp(
    X_orig_single,
    Y_orig_single,
    n_views,
    linear_slope_variance=0.1,
    linear_intercept_variance=0.1,
    noise_variance=0.01,
    rotation=True,
):

    n_samples_per_view = X_orig_single.shape[0]
    n_spatial_dims = X_orig_single.shape[1]

    kernel = rbf_covariance
    kernel_params_true = np.array([np.log(1.0), np.log(1.0)])

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

    X = X_orig.copy()

    # no_distortion_B = np.array([
    #         [0, np.pi * 0.5],
    #         [0, 0]
    #     ])

    for vv in range(n_views):

        # B = np.random.normal(
        #     loc=0,
        #     scale=np.sqrt(linear_slope_variance),
        #     size=(n_spatial_dims, n_spatial_dims),
        # )
        B = np.random.uniform(
            low=-linear_slope_variance,
            high=linear_slope_variance,
            size=(n_spatial_dims, n_spatial_dims),
        )
        polar_params = X_orig_single @ B
        r, theta = polar_params[:, 0], polar_params[:, 1]
        X_curr_view_warped = np.array(
            [
                X_orig_single[:, 0] + r * np.cos(theta),
                X_orig_single[:, 1] + r * np.sin(theta),
            ]
        ).T
        # import ipdb; ipdb.set_trace()
        # additive_warp = B[:, 0] * X_orig_single * np.vstack([np.cos(X_orig_single[:, 0] * B[0, 1]), np.sin(X_orig_single[:, 1] * B[1, 1])]).T
        # X_curr_view_warped = X_orig_single + additive_warp

        # X_curr_view_warped = X_orig_single @ curr_slopes + curr_intercepts
        X[
            n_samples_per_view * vv : n_samples_per_view * (vv + 1), :
        ] = X_curr_view_warped

    Y = np.concatenate([Y_orig_single] * n_views, axis=0)
    Y += np.random.normal(scale=np.sqrt(noise_variance), size=Y.shape)

    return X, Y, n_samples_list, view_idx


if __name__ == "__main__":

    pass
