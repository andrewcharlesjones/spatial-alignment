import numpy as np
import sys
import pandas as pd

sys.path.append("..")
from util import compute_distance
from warp_gp_multimodal import TwoLayerWarpGP

# from warp_gp_cca import TwoLayerWarpGP
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy

from gp_functions import rbf_covariance
from util import polar_warp
import torch

from tqdm import tqdm

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

device = "cuda" if torch.cuda.is_available() else "cpu"


def warp_example_experiment():

    n_views = 2
    n_genes = 10
    kernel = rbf_covariance
    kernel_params_true = np.array([1.0, 1.0])
    # n_samples_per_view = 30

    xlimits = [-10, 10]
    ylimits = [-10, 10]
    numticks = 6
    x1s = np.linspace(*xlimits, num=numticks)
    x2s = np.linspace(*ylimits, num=numticks)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig = np.vstack([X1.ravel(), X2.ravel()]).T
    n_samples_per_view = X_orig.shape[0]

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
    # X_orig = np.hstack(
    #     [
    #         np.random.uniform(low=-3, high=3, size=(n_samples_per_view, 1))
    #         for _ in range(2)
    #     ]
    # )
    Y_orig = np.vstack(
        [
            mvnpy.rvs(
                mean=np.zeros(n_samples_per_view),
                cov=kernel(X_orig, X_orig, kernel_params_true),
            )
            for _ in range(n_genes)
        ]
    ).T

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
        Y[view_idx[vv]] = curr_Y  # + np.random.normal(scale=0.1, size=curr_Y.shape)

    # Set up figure.
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    data_expression_ax = fig.add_subplot(121, frameon=False)
    latent_expression_ax = fig.add_subplot(122, frameon=False)
    plt.show(block=False)

    def callback(model):
        model.eval()
        markers = [".", "+", "^"]

        data_expression_ax.cla()
        latent_expression_ax.cla()
        data_expression_ax.set_title("Expression data")
        latent_expression_ax.set_title("G, Expression")

        for vv in range(n_views):
            data_expression_ax.scatter(
                X[view_idx[vv], 0],
                X[view_idx[vv], 1],
                c=np.sum(Y[view_idx[vv], :], axis=1),
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            latent_expression_ax.scatter(
                model.Gs["expression"].detach().numpy()[view_idx[vv], 0],
                model.Gs["expression"].detach().numpy()[view_idx[vv], 1],
                c=np.sum(Y[view_idx[vv], :], axis=1),
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
        plt.draw()
        plt.pause(1 / 60.0)

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }

    model = TwoLayerWarpGP(data_dict).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forwrard pass
        Gs, means_G_list, covs_G_list, means_Y, covs_Y = model.forward(
            {"expression": x}
        )

        # Compute loss
        loss = loss_fn(data_dict, Gs, means_G_list, covs_G_list, means_Y, covs_Y)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    N_EPOCHS = 2000
    PRINT_EVERY = 100
    loss_trace = []
    for t in range(N_EPOCHS):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)
        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            callback(model)

    plt.close()

    # import ipdb

    # ipdb.set_trace()

    X_warped = model.G.detach().numpy()

    plt.figure(figsize=(21, 7))
    markers = [".", "+", "^"]

    plt.subplot(131)
    plt.scatter(X_orig[:, 0], X_orig[:, 1], s=150, c=Y_orig[:, 0])
    plt.xlabel("Spatial dimension 1")
    plt.ylabel("Spatial dimension 2")
    plt.title(r"Ground truth")
    plt.colorbar()

    plt.subplot(132)
    for ii, curr_view_idx in enumerate(view_idx):
        curr_X = X[curr_view_idx]
        plt.scatter(
            curr_X[:, 0],
            curr_X[:, 1],
            label="View {}".format(ii + 1),
            marker=markers[ii],
            s=150,
            c=Y[curr_view_idx, 0],
        )
        plt.legend(loc="upper right")
        plt.xlabel("Spatial dimension 1")
        plt.ylabel("Spatial dimension 2")
        plt.title(r"Observed data space $\mathbf{X}$")
    plt.colorbar()

    plt.subplot(133)
    for ii, curr_view_idx in enumerate(view_idx):
        curr_X_warped = X_warped[curr_view_idx]
        plt.scatter(
            curr_X_warped[:, 0],
            curr_X_warped[:, 1],
            label="View {}".format(ii + 1),
            marker=markers[ii],
            s=150,
            c=Y[curr_view_idx, 0],
        )
        plt.legend(loc="upper right")
        plt.xlabel("Spatial dimension 1")
        plt.ylabel("Spatial dimension 2")
        plt.title(r"Reference space $\mathbf{G}$")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("./plots/example_alignment_simulated.png")
    plt.show()
    import ipdb

    ipdb.set_trace()


def one_d_experiment():

    n_views = 2
    n_genes = 1
    kernel = rbf_covariance
    kernel_params_true = np.array([np.log(1.0), np.log(1.0)])
    n_samples_per_view = 100

    X_orig = np.random.uniform(low=-5, high=5, size=(n_samples_per_view * n_views, 1))

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

    Y_orig = (
        np.vstack(
            [
                mvnpy.rvs(
                    mean=np.zeros(X_orig.shape[0]),
                    cov=kernel(X_orig, X_orig, kernel_params_true),
                )
                for _ in range(n_genes)
            ]
        ).T
        + np.random.normal(size=(X_orig.shape[0], n_genes), scale=0.1)
    )

    X = X_orig.copy()
    Y = Y_orig.copy()
    X[n_samples_per_view:] = X[n_samples_per_view:] * 1.1 + 0.5

    # Set up figure.
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    data_expression_ax = fig.add_subplot(211, frameon=False)
    latent_expression_ax = fig.add_subplot(212, frameon=False)
    plt.show(block=False)

    def callback(model):
        model.eval()
        markers = [".", "+", "^"]

        data_expression_ax.cla()
        latent_expression_ax.cla()
        data_expression_ax.set_title("Expression data")
        latent_expression_ax.set_title("G, Expression")

        for vv in range(n_views):
            data_expression_ax.scatter(
                X[view_idx[vv]],
                Y[view_idx[vv]],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            latent_expression_ax.scatter(
                model.Gs["expression"].detach().numpy()[view_idx[vv]],
                Y[view_idx[vv]],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
        plt.draw()
        plt.pause(1 / 60.0)

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }

    model = TwoLayerWarpGP(data_dict, n_spatial_dims=1, mean_penalty_param=10.0).to(
        device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forwrard pass
        Gs, means_G_list, covs_G_list, means_Y, covs_Y = model.forward(
            {"expression": x}
        )

        # Compute loss
        loss = loss_fn(data_dict, Gs, means_G_list, covs_G_list, means_Y, covs_Y)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    N_EPOCHS = 2000
    PRINT_EVERY = 100
    loss_trace = []
    for t in range(N_EPOCHS):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)
        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            callback(model)

    plt.close()

    # Set up figure.
    plt.figure(figsize=(14, 10))

    model.eval()
    markers = ["o", "^", "+"]

    data_expression_ax.cla()
    latent_expression_ax.cla()
    data_expression_ax.set_title("Data")
    latent_expression_ax.set_title("Aligned data")

    plt.subplot(211)
    plt.title("Data")
    plt.xlabel("Spatial coordinate")
    plt.ylabel("Output")
    for vv in range(n_views):
        plt.scatter(
            X[view_idx[vv]],
            Y[view_idx[vv]],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
        )
    plt.legend()
    plt.tight_layout()

    plt.subplot(212)
    plt.title("Aligned data")
    plt.xlabel("Spatial coordinate")
    plt.ylabel("Output")
    for vv in range(n_views):
        plt.scatter(
            model.Gs["expression"].detach().numpy()[view_idx[vv]],
            Y[view_idx[vv]],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
        )
    plt.legend()
    plt.tight_layout()

    plt.savefig("../plots/one_d_experiment.png")

    plt.show()
    import ipdb

    ipdb.set_trace()


def cross_experiment():
    font = {"size": 30}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True

    markers = [".", "+", "^", "o"]

    n_views = 3
    n_genes = 5
    kernel = rbf_covariance
    kernel_params_true = np.array([1.0, 1.0])
    # n_samples_per_view = 30

    xlimits = [-10, 10]
    ylimits = [-10, 10]
    numticks = 20
    x1s = np.linspace(*xlimits, num=numticks)
    x2s = np.linspace(*ylimits, num=numticks)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig = np.vstack([X1.ravel(), X2.ravel()]).T
    n_samples_per_view = X_orig.shape[0]

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
    # X_orig = np.hstack(
    #     [
    #         np.random.uniform(low=-3, high=3, size=(n_samples_per_view, 1))
    #         for _ in range(2)
    #     ]
    # )
    # Y_orig = np.vstack(
    #     [
    #         mvnpy.rvs(
    #             mean=np.zeros(n_samples_per_view),
    #             cov=kernel(X_orig, X_orig, kernel_params_true),
    #         )
    #         for _ in range(n_genes)
    #     ]
    # ).T

    colrange = xlimits[1] - xlimits[0]
    rowrange = ylimits[1] - ylimits[0]
    cross_collims = [xlimits[0] + colrange // 5 * 2, xlimits[0] + colrange // 5 * 3]
    cross_rowlims = [ylimits[0] + rowrange // 5 * 2, ylimits[0] + rowrange // 5 * 3]

    incross_col = np.logical_and(
        X_orig[:, 0] > cross_collims[0], X_orig[:, 0] < cross_collims[1]
    )
    incross_row = np.logical_and(
        X_orig[:, 1] > cross_rowlims[0], X_orig[:, 1] < cross_rowlims[1]
    )
    incross_idx = np.logical_or(incross_col, incross_row)

    Y_orig = np.zeros((X_orig.shape[0], n_genes))
    Y_orig[incross_idx, :] = 10 + np.random.normal(
        scale=3, size=Y_orig[incross_idx, :].shape
    )
    Y_orig[~incross_idx, :] = 0 + np.random.normal(
        scale=3, size=Y_orig[~incross_idx, :].shape
    )

    X = np.empty((np.sum(n_samples_list), 2))
    Y = np.empty((np.sum(n_samples_list), n_genes))

    plt.figure(figsize=(n_views * 7, 7))

    for vv in range(n_views):

        curr_X = X_orig.copy()

        # incross_col = np.logical_and(curr_X[:, 0] > cross_collims[0], curr_X[:, 0] < cross_collims[1])
        # incross_row = np.logical_and(curr_X[:, 1] > cross_rowlims[0], curr_X[:, 1] < cross_rowlims[1])
        # incross_idx = np.logical_or(incross_col, incross_row)

        # Warp
        linear_coeffs = np.random.normal(scale=0.1, size=2 * 2)
        rs_true, thetas_true = curr_X @ linear_coeffs[:2], curr_X @ linear_coeffs[2:]

        curr_X_observed = polar_warp(curr_X, rs_true, thetas_true)
        X[view_idx[vv]] = curr_X_observed

        curr_Y = Y_orig.copy()
        # curr_Y[incross_idx, :] = 10 + np.random.normal(scale=1, size=curr_Y[incross_idx, :].shape)
        # curr_Y[~incross_idx, :] = 0 + np.random.normal(scale=1, size=curr_Y[~incross_idx, :].shape)
        Y[view_idx[vv]] = curr_Y  # + np.random.normal(scale=0.1, size=curr_Y.shape)

        plt.subplot(1, n_views, vv + 1)
        plt.scatter(
            curr_X_observed[:, 0],
            curr_X_observed[:, 1],
            c=curr_Y[:, 0],
            s=200,
            marker="s",
        )
        plt.title("View {}".format(vv + 1))

    plt.savefig("../plots/cross_data.png")
    plt.close()
    # plt.show()
    # import ipdb; ipdb.set_trace()

    # Set up figure.
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    data_expression_ax = fig.add_subplot(121, frameon=False)
    latent_expression_ax = fig.add_subplot(122, frameon=False)
    plt.show(block=False)

    def callback(model):
        model.eval()

        data_expression_ax.cla()
        latent_expression_ax.cla()
        data_expression_ax.set_title("Expression data")
        latent_expression_ax.set_title("G, Expression")

        for vv in range(n_views):
            data_expression_ax.scatter(
                X[view_idx[vv], 0],
                X[view_idx[vv], 1],
                c=np.sum(Y[view_idx[vv], :], axis=1),
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=200,
            )
            latent_expression_ax.scatter(
                model.Gs["expression"].detach().numpy()[view_idx[vv], 0],
                model.Gs["expression"].detach().numpy()[view_idx[vv], 1],
                c=np.sum(Y[view_idx[vv], :], axis=1),
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=200,
            )
        plt.draw()
        plt.pause(1 / 60.0)

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }

    model = TwoLayerWarpGP(data_dict, distance_penalty_param=0.0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forwrard pass
        Gs, means_G_list, covs_G_list, means_Y, covs_Y = model.forward(
            {"expression": x}
        )

        # Compute loss
        loss = loss_fn(data_dict, Gs, means_G_list, covs_G_list, means_Y, covs_Y)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    N_EPOCHS = 400
    PRINT_EVERY = 25
    loss_trace = []
    for t in range(N_EPOCHS):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)
        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            callback(model)

    plt.close()

    # import ipdb

    # ipdb.set_trace()

    plt.figure(figsize=(35, 7))

    for vv in range(n_views):

        curr_idx = view_idx[vv]
        curr_X = X[curr_idx]
        curr_Y = Y[curr_idx]

        plt.subplot(1, n_views + 2, vv + 1)
        plt.scatter(
            curr_X[:, 0],
            curr_X[:, 1],
            c=curr_Y[:, 0],
            s=200,
            marker=markers[vv],
        )
        plt.title("View {}".format(vv + 1))

        plt.subplot(1, n_views + 2, n_views + 1)
        plt.scatter(
            curr_X[:, 0],
            curr_X[:, 1],
            c=curr_Y[:, 0],
            s=200,
            marker=markers[vv],
        )
    plt.subplot(1, n_views + 2, n_views + 1)
    plt.title("Overlaid views")

    # plt.savefig("../plots/cross_data.png")
    # plt.close()

    plt.subplot(1, n_views + 2, n_views + 2)

    for vv in range(n_views):

        curr_idx = view_idx[vv]
        curr_X = model.Gs["expression"][curr_idx].detach().numpy()
        curr_Y = Y[curr_idx]

        plt.scatter(
            curr_X[:, 0],
            curr_X[:, 1],
            c=curr_Y[:, 0],
            s=200,
            marker=markers[vv],
        )
    plt.title(r"Aligned coordinates $\mathbf{G}$")

    plt.savefig("../plots/cross_data_aligned.png")
    plt.show()
    plt.close()
    import ipdb

    ipdb.set_trace()


def n_genes_experiment():

    n_views = 2

    n_genes_list = [1, 5, 10, 20]
    kernel = rbf_covariance
    kernel_params_true = np.array([1.0, 1.0])
    # n_samples_per_view = 30

    xlimits = [-10, 10]
    ylimits = [-10, 10]
    numticks = 6
    x1s = np.linspace(*xlimits, num=numticks)
    x2s = np.linspace(*ylimits, num=numticks)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig = np.vstack([X1.ravel(), X2.ravel()]).T
    n_samples_per_view = X_orig.shape[0]

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
    # X_orig = np.hstack(
    #     [
    #         np.random.uniform(low=-3, high=3, size=(n_samples_per_view, 1))
    #         for _ in range(2)
    #     ]
    # )

    n_repeats = 5
    results = np.empty((n_repeats, len(n_genes_list)))

    for ii in tqdm(range(n_repeats)):
        for jj, n_genes in enumerate(n_genes_list):
            Y_orig = np.vstack(
                [
                    mvnpy.rvs(
                        mean=np.zeros(n_samples_per_view),
                        cov=kernel(X_orig, X_orig, kernel_params_true),
                    )
                    for _ in range(n_genes)
                ]
            ).T

            X = np.empty((np.sum(n_samples_list), 2))
            Y = np.empty((np.sum(n_samples_list), n_genes))

            for vv in range(n_views):

                curr_X = X_orig.copy()
                # Warp
                linear_coeffs = np.random.normal(scale=0.1, size=2 * 2)
                rs_true, thetas_true = (
                    curr_X @ linear_coeffs[:2],
                    curr_X @ linear_coeffs[2:],
                )

                curr_X_observed = polar_warp(curr_X, rs_true, thetas_true)
                X[view_idx[vv]] = curr_X_observed

                curr_Y = Y_orig.copy()
                Y[
                    view_idx[vv]
                ] = curr_Y  # + np.random.normal(scale=0.1, size=curr_Y.shape)

            # Set up figure.
            fig = plt.figure(figsize=(14, 7), facecolor="white")
            data_expression_ax = fig.add_subplot(121, frameon=False)
            latent_expression_ax = fig.add_subplot(122, frameon=False)
            plt.show(block=False)

            def callback(model):
                model.eval()
                markers = [".", "+", "^"]

                data_expression_ax.cla()
                latent_expression_ax.cla()
                data_expression_ax.set_title("Expression data")
                latent_expression_ax.set_title("G, Expression")

                for vv in range(n_views):
                    data_expression_ax.scatter(
                        X[view_idx[vv], 0],
                        X[view_idx[vv], 1],
                        c=np.sum(Y[view_idx[vv], :], axis=1),
                        label="View {}".format(vv + 1),
                        marker=markers[vv],
                        s=100,
                    )
                    latent_expression_ax.scatter(
                        model.Gs["expression"].detach().numpy()[view_idx[vv], 0],
                        model.Gs["expression"].detach().numpy()[view_idx[vv], 1],
                        c=np.sum(Y[view_idx[vv], :], axis=1),
                        label="View {}".format(vv + 1),
                        marker=markers[vv],
                        s=100,
                    )
                plt.draw()
                plt.pause(1 / 60.0)

            x = torch.from_numpy(X).float().clone()
            y = torch.from_numpy(Y).float().clone()
            data_dict = {
                "expression": {
                    "spatial_coords": x,
                    "outputs": y,
                    "n_samples_list": n_samples_list,
                }
            }

            model = TwoLayerWarpGP(data_dict).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

            def train(model, loss_fn, optimizer):
                model.train()

                # Forwrard pass
                Gs, means_G_list, covs_G_list, means_Y, covs_Y = model.forward(
                    {"expression": x}
                )

                # Compute loss
                loss = loss_fn(
                    data_dict, Gs, means_G_list, covs_G_list, means_Y, covs_Y
                )

                # Compute gradients and take optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                return loss.item()

            N_EPOCHS = 1000
            PRINT_EVERY = 50
            loss_trace = []
            for t in range(N_EPOCHS):
                loss = train(model, model.loss_fn, optimizer)
                loss_trace.append(loss)
                # if t % PRINT_EVERY == 0:
                #     print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
                #     callback(model)

            plt.close()

            fitted_coordinates = model.Gs["expression"].detach().numpy()
            view1_coords = fitted_coordinates[model.view_idx["expression"][0]]
            view2_coords = fitted_coordinates[model.view_idx["expression"][1]]

            mse = np.mean((view1_coords - view2_coords) ** 2)

            results[ii, jj] = mse

    results_df = pd.melt(pd.DataFrame(results, columns=n_genes_list))
    sns.lineplot(data=results_df, x="variable", y="value")
    plt.xlabel("Number of genes")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("../plots/n_genes_experiment.png")
    plt.show()

    import ipdb

    ipdb.set_trace()


def n_samples_experiment():

    n_views = 2

    n_samples_per_view_list = [1, 5, 10, 20]
    kernel = rbf_covariance
    kernel_params_true = np.array([1.0, 1.0])

    n_genes = 5
    n_repeats = 2
    results = np.empty((n_repeats, len(n_samples_per_view_list)))

    for ii in tqdm(range(n_repeats)):
        for jj, n_samples_per_view in enumerate(n_samples_per_view_list):

            X_orig = np.random.uniform(low=-10, high=10, size=(n_samples_per_view, 2))

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
            Y_orig = np.vstack(
                [
                    mvnpy.rvs(
                        mean=np.zeros(n_samples_per_view),
                        cov=kernel(X_orig, X_orig, kernel_params_true),
                    )
                    for _ in range(n_genes)
                ]
            ).T

            X = np.empty((np.sum(n_samples_list), 2))
            Y = np.empty((np.sum(n_samples_list), n_genes))

            for vv in range(n_views):

                curr_X = X_orig.copy()
                # Warp
                linear_coeffs = np.random.normal(scale=0.1, size=2 * 2)
                rs_true, thetas_true = (
                    curr_X @ linear_coeffs[:2],
                    curr_X @ linear_coeffs[2:],
                )

                curr_X_observed = polar_warp(curr_X, rs_true, thetas_true)
                X[view_idx[vv]] = curr_X_observed

                curr_Y = Y_orig.copy()
                Y[view_idx[vv]] = curr_Y + np.random.normal(
                    scale=0.1, size=curr_Y.shape
                )

            # Set up figure.
            fig = plt.figure(figsize=(14, 7), facecolor="white")
            data_expression_ax = fig.add_subplot(121, frameon=False)
            latent_expression_ax = fig.add_subplot(122, frameon=False)
            plt.show(block=False)

            def callback(model):
                model.eval()
                markers = [".", "+", "^"]

                data_expression_ax.cla()
                latent_expression_ax.cla()
                data_expression_ax.set_title("Expression data")
                latent_expression_ax.set_title("G, Expression")

                for vv in range(n_views):
                    data_expression_ax.scatter(
                        X[view_idx[vv], 0],
                        X[view_idx[vv], 1],
                        c=np.sum(Y[view_idx[vv], :], axis=1),
                        label="View {}".format(vv + 1),
                        marker=markers[vv],
                        s=100,
                    )
                    latent_expression_ax.scatter(
                        model.Gs["expression"].detach().numpy()[view_idx[vv], 0],
                        model.Gs["expression"].detach().numpy()[view_idx[vv], 1],
                        c=np.sum(Y[view_idx[vv], :], axis=1),
                        label="View {}".format(vv + 1),
                        marker=markers[vv],
                        s=100,
                    )
                plt.draw()
                plt.pause(1 / 60.0)

            x = torch.from_numpy(X).float().clone()
            y = torch.from_numpy(Y).float().clone()
            data_dict = {
                "expression": {
                    "spatial_coords": x,
                    "outputs": y,
                    "n_samples_list": n_samples_list,
                }
            }

            model = TwoLayerWarpGP(data_dict).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

            def train(model, loss_fn, optimizer):
                model.train()

                # Forwrard pass
                Gs, means_G_list, covs_G_list, means_Y, covs_Y = model.forward(
                    {"expression": x}
                )

                # Compute loss
                loss = loss_fn(
                    data_dict, Gs, means_G_list, covs_G_list, means_Y, covs_Y
                )

                # Compute gradients and take optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                return loss.item()

            N_EPOCHS = 2000
            PRINT_EVERY = 50
            loss_trace = []
            for t in range(N_EPOCHS):
                loss = train(model, model.loss_fn, optimizer)
                loss_trace.append(loss)
                # if t % PRINT_EVERY == 0:
                #     print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
                #     callback(model)

            plt.close()

            fitted_coordinates = model.Gs["expression"].detach().numpy()
            view1_coords = fitted_coordinates[model.view_idx["expression"][0]]
            view2_coords = fitted_coordinates[model.view_idx["expression"][1]]

            mse = np.mean((view1_coords - view2_coords) ** 2)
            # print("Error: {}".format(mse))
            # import ipdb

            # ipdb.set_trace()

            results[ii, jj] = mse

    results_df = pd.melt(pd.DataFrame(results, columns=n_samples_per_view_list))
    sns.lineplot(data=results_df, x="variable", y="value")
    plt.xlabel("Number of samples in each view")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("../plots/n_samples_experiment.png")
    plt.show()

    import ipdb

    ipdb.set_trace()


def toy_anatomy_experiment():
    n_views = 2
    n_genes = 4
    kernel = rbf_covariance
    kernel_params_true = np.array([1.0, 1.0])

    m_X_per_view = 30
    m_G = 30
    n_spatial_dims = 2
    n_latent_gps = n_genes
    # n_samples_per_view = 30

    xlimits = [-10, 10]
    ylimits = [-10, 10]
    numticks = 15
    x1s = np.linspace(*xlimits, num=numticks)
    x2s = np.linspace(*ylimits, num=numticks)
    X1, X2 = np.meshgrid(x1s, x2s)
    X_orig = np.vstack([X1.ravel(), X2.ravel()]).T
    n_samples_per_view = X_orig.shape[0]

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

    # X_orig = np.hstack(
    #     [
    #         np.random.uniform(low=-3, high=3, size=(n_samples_per_view, 1))
    #         for _ in range(2)
    #     ]
    # )
    # Y_orig = np.vstack(
    #     [
    #         mvnpy.rvs(
    #             mean=np.zeros(n_samples_per_view),
    #             cov=kernel(X_orig, X_orig, kernel_params_true),
    #         )
    #         for _ in range(n_genes)
    #     ]
    # ).T

    colrange = xlimits[1] - xlimits[0]
    rowrange = ylimits[1] - ylimits[0]
    cross_collims = [xlimits[0] + colrange // 5 * 2, xlimits[0] + colrange // 5 * 3]
    cross_rowlims = [ylimits[0] + rowrange // 5 * 2, ylimits[0] + rowrange // 5 * 3]

    incross_col = np.logical_and(
        X_orig[:, 0] > cross_collims[0], X_orig[:, 0] < cross_collims[1]
    )
    incross_row = np.logical_and(
        X_orig[:, 1] > cross_rowlims[0], X_orig[:, 1] < cross_rowlims[1]
    )
    incross_idx = np.logical_or(incross_col, incross_row)

    Y_orig = np.zeros((X_orig.shape[0], n_genes))
    gaussian_blob = mvnpy.pdf(
        X_orig, np.zeros(n_spatial_dims), 10 * np.eye(n_spatial_dims)
    )
    for jj in range(Y_orig.shape[1]):
        # Y_orig[:, jj] = gaussian_blob + np.random.normal(scale=0.001, size=gaussian_blob.shape)
        Y_orig[:, jj] = gaussian_blob + np.random.normal(scale=3)

    X = np.empty((np.sum(n_samples_list), 2))
    Y = np.empty((np.sum(n_samples_list), n_genes))

    for vv in range(n_views):

        curr_X = X_orig.copy()

        # Warp
        linear_coeffs = np.random.normal(scale=0.05, size=2 * 2)
        rs_true, thetas_true = curr_X @ linear_coeffs[:2], curr_X @ linear_coeffs[2:]

        curr_X_observed = polar_warp(curr_X, rs_true, thetas_true)
        X[view_idx[vv]] = curr_X_observed

        curr_Y = Y_orig.copy()
        Y[view_idx[vv]] = curr_Y  # + np.random.normal(scale=0.1, size=curr_Y.shape)

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }

    # model = VGPR(X, view_idx, n, n_spatial_dims, m_X_per_view=m_X_per_view, m_G=m_G).to(device)
    # model = VariationalWarpGP(data_dict, m_X_per_view=m_X_per_view, m_G=m_G).to(device)
    # model = VariationalWarpGP(
    #     data_dict,
    #     n_spatial_dims=n_spatial_dims,
    #     m_X_per_view=m_X_per_view,
    #     m_G=m_G,
    #     data_init=True,
    #     minmax_init=False,
    #     n_latent_gps=n_latent_gps,
    # ).to(device)

    model = TwoLayerWarpGP(
        data_dict,
        data_init=True,
        n_spatial_dims=2,
        # n_noise_variance_params=1,
        # kernel_func=gp.kernels.RBF,
        distance_penalty_param=0.0,
        mean_penalty_param=1.0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forwrard pass
        # G_samples, F_samples = model.forward({"expression": x})
        model.forward({"expression": x})

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
                data_expression_ax.scatter(
                    X[view_idx[vv], 0],
                    X[view_idx[vv], 1],
                    c=Y[view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
                )
                latent_expression_ax.scatter(
                    model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
                    model.G_means["expression"].detach().numpy()[view_idx[vv], 1],
                    c=Y[view_idx[vv], 0],
                    label="View {}".format(vv + 1),
                    marker=markers[vv],
                    s=100,
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

    N_EPOCHS = 4000
    PRINT_EVERY = 25
    loss_trace = []
    error_trace = []
    for t in range(N_EPOCHS):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)
        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            # F_latent_samples = model.forward({"expression": x})
            model.forward({"expression": x})
            callback(model)
            # curr_correlation = compute_warp_distance_correlation(true_coords=X_orig, estimated_coords=model.G_means["expression"].detach().numpy())
            error_trace.append(loss)
            # print(curr_correlation)

    print("Done!")

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":

    # warp_example_experiment()
    # one_d_experiment()
    cross_experiment()
    # n_genes_experiment()
    # n_samples_experiment()
    # warp_magnitude_experiment()
    # toy_anatomy_experiment()

    import ipdb

    ipdb.set_trace()
