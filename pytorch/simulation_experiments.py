import numpy as np
import sys

sys.path.append("..")
from util import compute_distance
from warp_gp_multimodal import TwoLayerWarpGP
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy

from gp_functions import rbf_covariance
from util import polar_warp
import torch


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


if __name__ == "__main__":

    warp_example_experiment()
    # n_genes_experiment()
    # n_samples_experiment()
    # warp_magnitude_experiment()

    import ipdb

    ipdb.set_trace()
