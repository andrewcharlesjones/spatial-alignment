import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.contrib.gp as gp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import pandas as pd
from scipy.stats import pearsonr

# from util import make_pinwheel
import seaborn as sns
from variational_warp_gp_lmc2 import VariationalWarpGP

# from variational_lmc_warp_gp_full import VariationalLMCWarpGP
# from variational_lmc_warp_gp_slim import VariationalLMCWarpGP
# from warp_gp_multimodal import TwoLayerWarpGP
# from variational_warp_gp import VariationalWarpGP
import sys

sys.path.append("..")
from gp_functions import rbf_covariance
from scipy.stats import multivariate_normal as mvnpy
from util import polar_warp

from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances

device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 50


def two_d_experiment():

    n_views = 2
    n_outputs = 5
    kernel = rbf_covariance
    kernel_params_true = np.array([np.log(1.0), np.log(1.0)])
    n_latent_gps = 10
    n_spatial_dims = 2

    xlimits = [-10, 10]
    ylimits = [-10, 10]
    numticks = 15
    x1s = np.linspace(*xlimits, num=numticks)
    x2s = np.linspace(*ylimits, num=numticks)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig_single = np.vstack([X1.ravel(), X2.ravel()]).T
    X_orig = np.concatenate([X_orig_single.copy(), X_orig_single.copy()], axis=0)
    n_samples_per_view = X_orig.shape[0] // 2

    m_G = 36
    m_X_per_view = 36

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

    Y_orig_latent = np.vstack(
        [
            mvnpy.rvs(
                mean=np.zeros(X_orig_single.shape[0]),
                cov=K_XX,
            )
            for _ in range(n_latent_gps)
        ]
    ).T

    W_mat = np.random.normal(size=(n_latent_gps, n_outputs))
    # W_mat = np.expand_dims(np.array([1, -1]), 0)
    Y_orig = Y_orig_latent @ W_mat
    # noise = np.random.normal(size=Y.shape, scale=0.1)
    # Y += noise

    # Y = Y_orig.copy()

    ## Warp the second view
    # X = X_orig.copy()
    # X[n_samples_per_view:] = X[n_samples_per_view:] * 1. + 0.5

    # X = np.empty((X_orig_single.shape[0] * n_views, 2))
    # Y = np.empty((X_orig_single.shape[0] * n_views, n_outputs))

    # for vv in range(n_views):

    #     curr_X = X_orig_single.copy()

    #     # Warp
    #     linear_coeffs = np.random.normal(scale=0.1, size=2 * 2)
    #     #
    #     rs_true, thetas_true = curr_X @ linear_coeffs[:2], curr_X @ linear_coeffs[2:]

    #     curr_X_observed = polar_warp(curr_X, rs_true, thetas_true)
    #     # import ipdb; ipdb.set_trace()
    #     X[view_idx[vv]] = curr_X_observed

    #     curr_Y = Y_orig.copy()
    #     Y[view_idx[vv]] = curr_Y #+ np.random.normal(scale=0.1, size=curr_Y.shape)

    # K_XX = kernel(X_orig_single, X_orig_single, [np.log(0.1), np.log(0.1)])
    # X1_warped = np.vstack(
    #     [mvnpy.rvs(X_orig_single[:, dd], K_XX) for dd in range(n_spatial_dims)]
    # ).T
    # X2_warped = np.vstack(
    #     [mvnpy.rvs(X_orig_single[:, dd], K_XX) for dd in range(n_spatial_dims)]
    # ).T
    # X = np.concatenate([X1_warped, X2_warped], axis=0)

    Y = np.concatenate([Y_orig, Y_orig], axis=0)
    X = X_orig.copy()
    X[n_samples_per_view:] = X[n_samples_per_view:] @ (
        np.eye(2) + np.random.normal(0, 0.05, size=(2, 2))
    )  # + np.random.normal(scale=0.25, size=X_orig_single.shape)
    X[:n_samples_per_view] = X[:n_samples_per_view] @ (
        np.eye(2) + np.random.normal(0, 0.05, size=(2, 2))
    )  # + np.random.normal(scale=0.25, size=X_orig_single.shape)

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()

    # rbf_kernel(x, x, 0., 0.)
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }
    # import ipdb; ipdb.set_trace()

    # model = VGPR(X, view_idx, n, n_spatial_dims, m_X_per_view=m_X_per_view, m_G=m_G).to(device)
    # model = VariationalLMCWarpGP(
    #     data_dict,
    #     n_spatial_dims=n_spatial_dims,
    #     m_X_per_view=m_X_per_view,
    #     m_G=m_G,
    #     data_init=True,
    #     minmax_init=False,
    #     n_latent_gps=n_latent_gps,
    # ).to(device)
    model = VariationalWarpGP(
        data_dict,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=False,
        minmax_init=False,
        grid_init=True,
        n_latent_gps=n_latent_gps,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forwrard pass
        G_samples, F_samples = model.forward({"expression": x})

        # Compute loss
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(21, 7), facecolor="white")
    if n_spatial_dims == 2:
        data_expression_ax = fig.add_subplot(131, frameon=False)
        latent_expression_ax = fig.add_subplot(132, frameon=False)
        error_ax = fig.add_subplot(133, frameon=False)
    elif n_spatial_dims == 1:
        data_expression_ax = fig.add_subplot(311, frameon=False)
        latent_expression_ax = fig.add_subplot(312, frameon=False)
        error_ax = fig.add_subplot(313, frameon=False)
    plt.show(block=False)

    def callback(model):
        model.eval()
        markers = [".", "+", "^"]
        colors = ["blue", "orange"]

        data_expression_ax.cla()
        latent_expression_ax.cla()
        data_expression_ax.set_title("Observed data")
        latent_expression_ax.set_title("Aligned data")

        for vv in range(n_views):

            error_ax.plot(error_trace, c="black")

            if n_spatial_dims == 2:
                # import ipdb; ipdb.set_trace()
                data_expression_ax.scatter(
                    X[view_idx[vv], 0],
                    X[view_idx[vv], 1],
                    c=Y[view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=200,
                )
                latent_expression_ax.scatter(
                    model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
                    model.G_means["expression"].detach().numpy()[view_idx[vv], 1],
                    c=Y[view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=200,
                )
                latent_expression_ax.scatter(
                    model.Xtilde.detach().numpy()[vv][:, 0],
                    model.Xtilde.detach().numpy()[vv][:, 1],
                    c="red",
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=200,
                )

                # latent_expression_ax.scatter(
                #     model.Xtilde.detach().numpy()[vv][:, 0],
                #     model.Xtilde.detach().numpy()[vv][:, 1],
                #     label="View {}".format(vv + 1),
                #     marker=markers[vv],
                #     s=100,
                #     c="red",
                # )
                # latent_expression_ax.scatter(
                #     model.Gtilde.detach().numpy()[:, 0],
                #     model.Gtilde.detach().numpy()[:, 1],
                #     label="View {}".format(vv + 1),
                #     marker=markers[vv],
                #     s=100,
                #     c="pink",
                # )

            elif n_spatial_dims == 1:
                data_expression_ax.scatter(
                    X[view_idx[vv], 0],
                    Y[view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                    c="blue",
                )
                data_expression_ax.scatter(
                    X[view_idx[vv], 0],
                    Y[view_idx[vv], 1],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                    c="orange",
                )
                latent_expression_ax.scatter(
                    model.G_means.detach().numpy()[view_idx[vv], 0],
                    Y[view_idx[vv], 0],
                    c="blue",
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                )
                latent_expression_ax.scatter(
                    model.G_means.detach().numpy()[view_idx[vv], 0],
                    Y[view_idx[vv], 1],
                    c="orange",
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                )
        plt.draw()
        plt.pause(1 / 60.0)

    N_EPOCHS = 3000
    PRINT_EVERY = 25
    loss_trace = []
    error_trace = []
    for t in range(N_EPOCHS):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)
        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            F_latent_samples = model.forward({"expression": x})
            callback(model)
            error_trace.append(loss)

    print("Done!")

    plt.close()

    import matplotlib

    font = {"size": LATEX_FONTSIZE}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(24, 10))  # , facecolor="white")
    data_expression_ax = fig.add_subplot(121, frameon=False)
    latent_expression_ax = fig.add_subplot(122, frameon=False)
    data_expression_ax.set_xlabel("Spatial coordinate 1")
    latent_expression_ax.set_xlabel("Spatial coordinate 1")
    data_expression_ax.set_ylabel("Spatial coordinate 2")
    latent_expression_ax.set_ylabel("Spatial coordinate 2")
    data_expression_ax.set_title("Observed")
    latent_expression_ax.set_title("Aligned")

    markers = [".", "+", "^"]
    colors = ["blue", "orange"]
    for vv in range(n_views):
        data_expression_ax.scatter(
            X[view_idx[vv], 0],
            X[view_idx[vv], 1],
            c=Y[view_idx[vv], 0],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=200,
        )
        latent_expression_ax.scatter(
            model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
            model.G_means["expression"].detach().numpy()[view_idx[vv], 1],
            c=Y[view_idx[vv], 0],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=200,
        )
    latent_expression_ax.legend(bbox_to_anchor=(1.1, 1.05), fontsize=40)
    plt.tight_layout()
    plt.savefig("../plots/two_d_simulation.png")
    plt.show()

    import ipdb

    ipdb.set_trace()


def one_d_experiment():

    n_views = 2
    n_outputs = 5
    kernel = rbf_covariance
    kernel_params_true = np.array([np.log(1.0), np.log(1.0)])
    n_latent_gps = 2
    n_spatial_dims = 1

    n_samples_per_view = 100
    X_orig_single = np.random.uniform(-10, 10, size=(n_samples_per_view, 1))
    X_orig = np.concatenate([X_orig_single.copy(), X_orig_single.copy()], axis=0)

    m_G = 9
    m_X_per_view = 9

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

    Y_orig = np.vstack(
        [
            mvnpy.rvs(
                mean=np.zeros(X_orig_single.shape[0]),
                cov=kernel(X_orig_single, X_orig_single, kernel_params_true),
            )
            for _ in range(n_outputs)
        ]
    ).T

    # W_mat = np.random.normal(size=(n_latent_gps, n_outputs))
    # # W_mat = np.expand_dims(np.array([1, -1]), 0)
    # Y_orig = Y_orig_latent @ W_mat

    Y = np.concatenate([Y_orig, Y_orig], axis=0)
    X = X_orig.copy()
    X[n_samples_per_view:] = X[n_samples_per_view:] * 1.1 + 0.3

    # import ipdb; ipdb.set_trace()

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()

    # rbf_kernel(x, x, 0., 0.)
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }

    # model = VGPR(X, view_idx, n, n_spatial_dims, m_X_per_view=m_X_per_view, m_G=m_G).to(device)
    # model = VariationalLMCWarpGP(
    #     data_dict,
    #     n_spatial_dims=n_spatial_dims,
    #     m_X_per_view=m_X_per_view,
    #     m_G=m_G,
    #     data_init=True,
    #     minmax_init=False,
    #     n_latent_gps=n_latent_gps,
    # ).to(device)
    model = VariationalWarpGP(
        data_dict,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=False,
        minmax_init=False,
        grid_init=True,
        # n_latent_gps=n_latent_gps,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forwrard pass
        G_samples, F_samples = model.forward({"expression": x})

        # Compute loss
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(21, 7), facecolor="white")
    if n_spatial_dims == 2:
        data_expression_ax = fig.add_subplot(131, frameon=False)
        latent_expression_ax = fig.add_subplot(132, frameon=False)
        error_ax = fig.add_subplot(133, frameon=False)
    elif n_spatial_dims == 1:
        data_expression_ax = fig.add_subplot(311, frameon=False)
        latent_expression_ax = fig.add_subplot(312, frameon=False)
        error_ax = fig.add_subplot(313, frameon=False)
    plt.show(block=False)

    def callback(model):
        model.eval()
        markers = [".", "+", "^"]
        colors = ["blue", "orange"]

        data_expression_ax.cla()
        latent_expression_ax.cla()
        data_expression_ax.set_title("Observed data")
        latent_expression_ax.set_title("Aligned data")

        for vv in range(n_views):

            error_ax.plot(error_trace, c="black")

            if n_spatial_dims == 2:
                # import ipdb; ipdb.set_trace()
                data_expression_ax.scatter(
                    X[view_idx[vv], 0],
                    X[view_idx[vv], 1],
                    c=Y[view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=200,
                )
                latent_expression_ax.scatter(
                    model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
                    model.G_means["expression"].detach().numpy()[view_idx[vv], 1],
                    c=Y[view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=200,
                )
                latent_expression_ax.scatter(
                    model.Xtilde.detach().numpy()[vv][:, 0],
                    model.Xtilde.detach().numpy()[vv][:, 1],
                    c="red",
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=200,
                )

                # latent_expression_ax.scatter(
                #     model.Xtilde.detach().numpy()[vv][:, 0],
                #     model.Xtilde.detach().numpy()[vv][:, 1],
                #     label="View {}".format(vv + 1),
                #     marker=markers[vv],
                #     s=100,
                #     c="red",
                # )
                # latent_expression_ax.scatter(
                #     model.Gtilde.detach().numpy()[:, 0],
                #     model.Gtilde.detach().numpy()[:, 1],
                #     label="View {}".format(vv + 1),
                #     marker=markers[vv],
                #     s=100,
                #     c="pink",
                # )

            elif n_spatial_dims == 1:
                data_expression_ax.scatter(
                    X[view_idx[vv], 0],
                    Y[view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                    c="blue",
                )
                data_expression_ax.scatter(
                    X[view_idx[vv], 0],
                    Y[view_idx[vv], 1],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                    c="orange",
                )
                latent_expression_ax.scatter(
                    model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
                    Y[view_idx[vv], 0],
                    c="blue",
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                )
                latent_expression_ax.scatter(
                    model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
                    Y[view_idx[vv], 1],
                    c="orange",
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                )
        plt.draw()
        plt.pause(1 / 60.0)

    N_EPOCHS = 300
    PRINT_EVERY = 50
    loss_trace = []
    error_trace = []
    for t in range(N_EPOCHS):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)
        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            F_latent_samples = model.forward({"expression": x})
            # import ipdb; ipdb.set_trace()
            callback(model)
            # curr_correlation = compute_warp_distance_correlation(true_coords=X_orig, estimated_coords=model.G_means["expression"].detach().numpy())
            error_trace.append(loss)
            # print(model.Gtilde)
            # print(curr_correlation)
            # import ipdb; ipdb.set_trace()
            # print(model.kernel_G_view0._parameters['lengthscale_unconstrained'].detach().numpy())

    print("Done!")

    plt.close()

    import matplotlib

    font = {"size": LATEX_FONTSIZE}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(10, 10))  # , facecolor="white")
    data_expression_ax = fig.add_subplot(211, frameon=False)
    latent_expression_ax = fig.add_subplot(212, frameon=False)
    data_expression_ax.set_xlabel("Spatial coordinate")
    latent_expression_ax.set_xlabel("Spatial coordinate")
    data_expression_ax.set_ylabel("Outcome")
    latent_expression_ax.set_ylabel("Outcome")
    data_expression_ax.set_title("Observed")
    latent_expression_ax.set_title("Aligned")

    markers = [".", "+", "^"]
    colors = ["blue", "orange"]
    for vv in range(n_views):
        data_expression_ax.scatter(
            X[view_idx[vv], 0],
            Y[view_idx[vv], 0],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
            c="blue",
        )
        data_expression_ax.scatter(
            X[view_idx[vv], 0],
            Y[view_idx[vv], 1],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
            c="orange",
        )
        latent_expression_ax.scatter(
            model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
            Y[view_idx[vv], 0],
            c="blue",
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
        )
        latent_expression_ax.scatter(
            model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
            Y[view_idx[vv], 1],
            c="orange",
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
        )
    # data_expression_ax.legend()
    plt.tight_layout()
    plt.savefig("../plots/one_d_simulation.png")
    plt.show()

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":

    two_d_experiment()
    # one_d_experiment()
    # toy_anatomy_experiment()
