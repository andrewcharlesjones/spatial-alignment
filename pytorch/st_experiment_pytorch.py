import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy
from scipy.stats import multivariate_normal as mvno
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import socket

sys.path.append("..")
from util import get_st_coordinates, polar_warp
# from warp_gp_multimodal import TwoLayerWarpGP
from variational_warp_gp_lmc2 import VariationalWarpGP
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy


from gp_functions import rbf_covariance
from util import polar_warp

from tqdm import tqdm

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

N_EPOCHS = 4000
PRINT_EVERY = 25
N_LATENT_GPS = 3
MEAN_PENALTY_PARAM = 1e0


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    if socket.gethostname() == "andyjones":
        layer2_path = "../data/st/layer2.csv"
    else:
        layer2_path = "/tigress/aj13/spatial_data/st/layer2.csv"

    layer2_raw_df = pd.read_csv(layer2_path, index_col=0)
    X_orig = get_st_coordinates(layer2_raw_df)
    X_orig -= X_orig.min(0)
    X_orig /= X_orig.max(0)
    X_orig *= 20

    n, n_genes_total = layer2_raw_df.shape

    kernel = rbf_covariance

    ## Select high-variance genes
    n_genes = 25
    gene_vars = layer2_raw_df.var(0).values
    sorted_idx = np.argsort(-gene_vars)
    chosen_idx = sorted_idx[:n_genes]
    gene_names = layer2_raw_df.columns.values[chosen_idx]
    Y_orig_unstdized = np.log(layer2_raw_df.values[:, chosen_idx] + 1)

    assert X_orig.shape[0] == Y_orig_unstdized.shape[0]

    ## Standardize expression
    Y_orig = (Y_orig_unstdized - Y_orig_unstdized.mean(0)) / Y_orig_unstdized.std(0)

    ## Artificially split into two datasets
    n_views = 2
    # n1 = n // 2
    # n2 = n - n1
    n1, n2 = 40, 40
    # n1, n2 = X_orig.shape[0], X_orig.shape[0]
    n_total = n1 + n2
    p = 2
    # import ipdb; ipdb.set_trace()
    # data1_idx = np.random.choice(np.arange(n), size=n1, replace=False)
    # data2_idx = np.random.choice(np.arange(n), size=n2, replace=False)
    data1_idx = np.arange(n)[:n1]
    data2_idx = np.arange(n)[:n2]
    view_idx = np.array([np.arange(0, n1), np.arange(n1, n1 + n2)])
    n_samples_list = [n1, n2]

    # X1 = X_orig
    # X2 = X_orig
    # Y1 = Y_orig
    # Y2 = Y_orig
    X1 = X_orig[data1_idx, :]
    X2 = X_orig[data2_idx, :]
    Y1 = Y_orig[data1_idx, :]
    Y2 = Y_orig[data2_idx, :]
    Y = np.concatenate([Y1, Y2], axis=0)

    ## Warp coordinates
    linear_coeffs = np.random.normal(scale=0.05, size=n_views * 2 * 2)
    r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
    r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

    X1_observed = polar_warp(X1, r1s_true, theta1s_true)
    X2_observed = polar_warp(X2, r2s_true, theta2s_true)
    X = np.vstack([X1_observed, X2_observed])
    # pd.DataFrame(X).to_csv("./out/intermediate_st_X.csv", index=False)
    # pd.DataFrame(Y).to_csv("./out/intermediate_st_Y.csv", index=False)

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
                model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
                model.G_means["expression"].detach().numpy()[view_idx[vv], 1],
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

    # model = TwoLayerWarpGP(data_dict).to(device)
    m_X_per_view = 36
    m_G = 36
    model = VariationalWarpGP(
        data_dict,
        n_spatial_dims=2,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=False,
        minmax_init=False,
        grid_init=True,
        n_latent_gps=N_LATENT_GPS,
        mean_penalty_param=MEAN_PENALTY_PARAM,
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forwrard pass
        # Gs, means_G_list, covs_G_list, means_Y, covs_Y = model.forward(
        #     {"expression": x}
        # )
        G_samples, F_samples = model.forward(
            {"expression": x}
        )

        # Compute loss
        # loss = loss_fn(data_dict, Gs, means_G_list, covs_G_list, means_Y, covs_Y)
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    
    loss_trace = []
    for t in range(N_EPOCHS):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)
        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            callback(model)
            print(np.exp(model.kernel_variances.detach().numpy()))

    plt.close()

    # import ipdb

    # ipdb.set_trace()

    X_warped = model.G_means["expression"].detach().numpy()

    plt.figure(figsize=(21, 7))
    markers = [".", "+", "^"]

    plt.subplot(131)
    plt.scatter(
        X_orig[data1_idx, 0], X_orig[data1_idx, 1], s=150, c=Y_orig[data1_idx, 0]
    )
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
    plt.savefig("../plots/example_alignment_st_reference_fixed.png")
    plt.show()

    import ipdb

    ipdb.set_trace()
