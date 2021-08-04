import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy
from scipy.stats import multivariate_normal as mvno
import pandas as pd
from gp_functions import rbf_covariance
from warp_gp_pytorch import TwoLayerWarpGP, loss_fn, SpatialDataset
from util import get_st_coordinates, polar_warp
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import socket

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    if socket.gethostname() == "andyjones":
        layer2_path = "./data/st/layer2.csv"
    else:
        layer2_path = "/tigress/aj13/spatial_data/st/layer2.csv"

    layer2_raw_df = pd.read_csv(layer2_path, index_col=0)
    X_orig = get_st_coordinates(layer2_raw_df)
    X_orig -= X_orig.min(0)
    X_orig /= X_orig.max(0)

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
    n1, n2 = 30, 30
    n_total = n1 + n2
    p = 2
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
    pd.DataFrame(X).to_csv("./out/intermediate_st_X.csv", index=False)
    pd.DataFrame(Y).to_csv("./out/intermediate_st_Y.csv", index=False)

    model = TwoLayerWarpGP(
        n_views=n_views,
        n_samples_list=n_samples_list,
        n_features=n_genes,
        G_init=torch.from_numpy(X).clone().float(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = SpatialDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    train_dataloader = DataLoader(
        dataset, batch_size=dataset.__len__(), shuffle=False, num_workers=0
    )

    def train(dataloader, model, loss_fn, optimizer):
        model.train()
        size = len(dataloader.dataset)
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            # Compute prediction error
            G, mean_G, cov_G_list, mean_Y, cov_Y = model.forward(x)
            loss = loss_fn(x, y, G, mean_G, cov_G_list, mean_Y, cov_Y, view_idx)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    data_ax = fig.add_subplot(121, frameon=False)
    latent_ax = fig.add_subplot(122, frameon=False)
    plt.show(block=False)

    def callback(model):
        model.eval()
        markers = [".", "+", "^"]

        data_ax.cla()
        latent_ax.cla()
        data_ax.set_title("Data")
        latent_ax.set_title("G")

        for vv in range(n_views):
            data_ax.scatter(
                X[view_idx[vv], 0],
                X[view_idx[vv], 1],
                c=Y[view_idx[vv], 0],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            latent_ax.scatter(
                model.G.detach().numpy()[view_idx[vv], 0],
                model.G.detach().numpy()[view_idx[vv], 1],
                c=Y[view_idx[vv], 0],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            # latent_ax.set_xlim([np.min(X[:, 0]), np.max(X[:, 0])])
            # latent_ax.set_ylim([np.min(X[:, 1]), np.max(X[:, 1])])

        pd.DataFrame(model.G.detach().numpy()).to_csv(
            "./out/intermediate_st_G.csv", index=False
        )
        plt.savefig("./plots/intermediate_st_results.png")
        plt.draw()
        plt.pause(1.0 / 60.0)

    epochs = 2000
    loss_trace = []
    for t in range(epochs):
        loss = train(train_dataloader, model, loss_fn, optimizer)
        loss_trace.append(loss)
        if t % 25 == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            callback(model)
            # print(model.G)
    print("Done!")
    plt.close()
    plt.plot(loss_trace)
    plt.savefig("./plots/loss_trace_st.png")
    plt.close()

    # import ipdb

    # ipdb.set_trace()

    X_warped = model.G.detach().numpy()

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
    plt.savefig("./plots/example_alignment_st_reference_fixed.png")
    plt.show()

    import ipdb

    ipdb.set_trace()
