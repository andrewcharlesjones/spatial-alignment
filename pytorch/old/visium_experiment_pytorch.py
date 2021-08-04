import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy
from scipy.stats import multivariate_normal as mvno
import pandas as pd
from warp_gp_histology_pytorch import TwoLayerWarpHistologyGP, loss_fn, SpatialDataset
import sys

sys.path.append("..")
from util import get_st_coordinates, polar_warp
import torch
from torch.utils.data import Dataset, DataLoader
import socket
from os.path import join as pjoin
from sklearn.gaussian_process.kernels import RBF

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

N_VIEWS = 2
N_SAMPLES = 40
N_GENES = 10
N_HISTOLOGY_CHANNELS = 3
SCALEFACTOR = 0.17211704
GRAY_PIXEL_VAL = 0.7
WARP_SCALE = 0.1
PRINT_EVERY = 100
LEARNING_RATE = 1e-3
N_EPOCHS = 4000

DATA_DIR = "../data/visium/mouse_brain/sample1"

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":

    ## Load data
    spatial_locs_sample1_path = pjoin(
        DATA_DIR, "filtered_feature_bc_matrix/spatial_locs_small.csv"
    )
    data_sample1_path = pjoin(
        DATA_DIR, "filtered_feature_bc_matrix/gene_expression_small.csv"
    )
    X_df_sample1 = pd.read_csv(spatial_locs_sample1_path, index_col=0)
    Y_df_sample1 = pd.read_csv(data_sample1_path, index_col=0)
    X_df_sample1["x_position"] = X_df_sample1.row_pxl * SCALEFACTOR
    X_df_sample1["y_position"] = X_df_sample1.col_pxl * SCALEFACTOR

    ## Make second sample a copy of sample1 for now
    X_df_sample2 = X_df_sample1.copy()
    Y_df_sample2 = Y_df_sample1.copy()
    X_df_sample2["x_position"] = X_df_sample2.row_pxl * SCALEFACTOR
    X_df_sample2["y_position"] = X_df_sample2.col_pxl * SCALEFACTOR

    ## Load histology image
    tissue_image_hires_sample1 = plt.imread(
        pjoin(DATA_DIR, "spatial/tissue_hires_image.png")
    )

    ## Make box around histology image part that has expression reads
    image_xlims = np.array(
        [int(X_df_sample1.x_position.min()), int(X_df_sample1.x_position.max())]
    )
    image_ylims = np.array(
        [int(X_df_sample1.y_position.min()), int(X_df_sample1.y_position.max())]
    )
    tissue_img_cropped_sample1 = tissue_image_hires_sample1[
        image_ylims[0] : image_ylims[1], image_xlims[0] : image_xlims[1]
    ]

    ## Convert image pixel coordinates into (x, y) coordinates
    x1_img, x2_img = np.meshgrid(
        np.arange(tissue_img_cropped_sample1.shape[1]),
        np.arange(tissue_img_cropped_sample1.shape[0]),
    )
    X_img = np.vstack([x1_img.ravel(), x2_img.ravel()]).T

    ## Adjust expression spatial coordinates accordingly
    X_df_sample1["x_position"] -= image_xlims[0]
    X_df_sample1["y_position"] -= image_ylims[0]
    X_df_sample2["x_position"] -= image_xlims[0]
    X_df_sample2["y_position"] -= image_ylims[0]

    pixel_vals = np.array(
        [
            tissue_img_cropped_sample1[X_img[ii, 1], X_img[ii, 0]]
            for ii in range(X_img.shape[0])
        ]
    )

    ## Remove gray border on image
    nongray_idx = np.where(~np.all(np.round(pixel_vals, 1) == GRAY_PIXEL_VAL, axis=1))[
        0
    ]
    pixel_vals = pixel_vals[nongray_idx, :]
    X_img = X_img[nongray_idx, :]
    X_img_original = X_img.copy()
    pixel_vals_original = pixel_vals.copy()

    ## Subsample pixels
    rand_idx = np.random.choice(
        np.arange(X_img.shape[0]), size=N_SAMPLES, replace=False
    )
    X_img = X_img[rand_idx, :]
    pixel_vals = pixel_vals[rand_idx, :]

    ##### Prep expression #####
    X_orig_sample1 = X_df_sample1[["x_position", "y_position"]]
    X_orig_sample2 = X_df_sample2[["x_position", "y_position"]]

    X1_expression = X_orig_sample1.values
    X2_expression = X_orig_sample2.values

    assert np.all(X_df_sample1.index.values == Y_df_sample1.index.values)
    assert np.all(X_df_sample2.index.values == Y_df_sample2.index.values)

    ## Select high-variance genes (columns are already sorted here)
    chosen_idx = np.arange(N_GENES)
    gene_names = Y_df_sample1.columns.values[chosen_idx]
    Y_orig_unstdized_sample1 = Y_df_sample1.values[:, chosen_idx]
    Y_orig_unstdized_sample2 = Y_df_sample2.values[:, chosen_idx]

    ## Standardize expression
    Y1_expression = (
        Y_orig_unstdized_sample1 - Y_orig_unstdized_sample1.mean(0)
    ) / Y_orig_unstdized_sample1.std(0)
    Y2_expression = (
        Y_orig_unstdized_sample2 - Y_orig_unstdized_sample2.mean(0)
    ) / Y_orig_unstdized_sample2.std(0)

    assert X_orig_sample1.shape[0] == Y1_expression.shape[0]
    assert X_orig_sample2.shape[0] == Y2_expression.shape[0]

    ## Subsample expression locations
    n1_expression = X1_expression.shape[0]
    expression1_subsample_idx = np.random.choice(
        np.arange(n1_expression), size=N_SAMPLES, replace=False
    )
    X1_expression = X1_expression[expression1_subsample_idx]
    X2_expression = X2_expression[expression1_subsample_idx]
    Y1_expression = Y1_expression[expression1_subsample_idx]
    Y2_expression = Y2_expression[expression1_subsample_idx]

    ##### Histology #####
    X1_histology = X_img
    X2_histology = X_img
    Y1_histology = pixel_vals
    Y2_histology = pixel_vals

    assert X1_histology.shape[0] == Y1_histology.shape[0]
    assert X2_histology.shape[0] == Y2_histology.shape[0]

    n1_expression, n2_expression = Y1_expression.shape[0], Y2_expression.shape[0]
    n1_histology, n2_histology = Y1_histology.shape[0], Y2_histology.shape[0]

    ## Get the indices of each view in each modality
    view_idx_expression = np.array(
        [
            np.arange(0, n1_expression),
            np.arange(n1_expression, n1_expression + n2_expression),
        ]
    )
    view_idx_histology = np.array(
        [
            np.arange(0, n1_histology),
            np.arange(n1_histology, n1_histology + n2_histology),
        ]
    )
    n_samples_list_expression = [n1_expression, n2_expression]
    n_samples_list_histology = [n1_histology, n2_histology]

    # Divide by max val
    max_val = max(
        np.max(X1_expression),
        np.max(X1_histology),
        np.max(X2_expression),
        np.max(X2_histology),
    )
    X1 = np.concatenate([X1_expression, X1_histology]) / max_val
    X2 = np.concatenate([X2_expression, X2_histology]) / max_val

    ## Warp coordinates
    linear_coeffs = np.random.normal(scale=WARP_SCALE, size=N_VIEWS * 2 * 2)
    # linear_coeffs = np.concatenate([[-0.2] * 4, [0.2] * 4])
    r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
    r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

    X1_observed = polar_warp(X1, r1s_true, theta1s_true)
    X2_observed = polar_warp(X2, r2s_true, theta2s_true)

    # kernel = RBF(length_scale=1e-6)
    # kernel_func = lambda x, y: 0.001 * kernel(x, y)
    # noise = 1e-5
    # K_X1X1 = kernel_func(X1, X1)
    # K_X2X2 = kernel_func(X2, X2)
    # X11_observed = mvnpy.rvs(mean=X1[:, 0], cov=K_X1X1 * np.eye(X1.shape[0]))
    # X12_observed = mvnpy.rvs(X1[:, 1], K_X1X1 + noise * np.eye(X1.shape[0]))
    # X1_observed = np.vstack([X11_observed, X12_observed]).T
    # X21_observed = mvnpy.rvs(X2[:, 0], K_X2X2 + noise * np.eye(X2.shape[0]))
    # X22_observed = mvnpy.rvs(X2[:, 1], K_X2X2 + noise * np.eye(X2.shape[0]))
    # X2_observed = np.vstack([X21_observed, X22_observed]).T

    X_expression = np.concatenate(
        [X1_observed[: X1_expression.shape[0]], X2_observed[: X2_expression.shape[0]]]
    )
    X_histology = np.concatenate(
        [X1_observed[X1_expression.shape[0] :], X2_observed[X2_expression.shape[0] :]]
    )

    Y_expression = np.concatenate([Y1_expression, Y2_expression], axis=0)
    Y_histology = np.concatenate([Y1_histology, Y2_histology], axis=0)

    # plt.figure(figsize=(21, 14))
    # plt.subplot(231)
    # rand_idx = np.random.choice(np.arange(X_img_original.shape[0]), size=10000, replace=False)
    # rand_idx = np.sort(rand_idx)
    # plt.scatter(X_img_original[rand_idx, 0], X_img_original[rand_idx, 1], c=pixel_vals_original[rand_idx, :])
    # plt.axis('off')
    # plt.title("Original sample")
    # plt.subplot(232)
    # plt.scatter(X_histology[:N_SAMPLES, 0], X_histology[:N_SAMPLES, 1], c=Y_histology[:N_SAMPLES, :])
    # plt.axis('off')
    # plt.title("Warped sample 1")
    # plt.subplot(233)
    # plt.scatter(X_histology[N_SAMPLES:, 0], X_histology[N_SAMPLES:, 1], c=Y_histology[N_SAMPLES:, :])
    # plt.axis('off')
    # plt.title("Warped sample 2")

    # plt.subplot(234)
    # plt.scatter(X_df_sample1["x_position"], X_df_sample1["y_position"], c=Y_df_sample1.sum(1))
    # plt.axis('off')
    # plt.subplot(235)
    # plt.scatter(X_expression[:N_SAMPLES, 0], X_expression[:N_SAMPLES, 1], c=Y_expression[:N_SAMPLES].sum(1))
    # plt.axis('off')
    # plt.subplot(236)
    # plt.scatter(X_expression[N_SAMPLES:, 0], X_expression[N_SAMPLES:, 1], c=Y_expression[N_SAMPLES:].sum(1))
    # plt.axis('off')
    # plt.savefig("../plots/warped_visium_example.png")
    # plt.show()
    # import ipdb; ipdb.set_trace()

    data_dict = {
        "expression": {
            "spatial_coords": x_expression,
            "outputs": y_expression,
            "n_samples_list": n_samples_list_expression,
        },
        "histology": {
            "spatial_coords": x_histology,
            "outputs": y_histology,
            "n_samples_list": n_samples_list_histology,
        },
    }

    ## Prep model
    model = TwoLayerWarpGP(data_dict).to(device)

    ## Prep model
    model = TwoLayerWarpHistologyGP(
        n_views=N_VIEWS,
        n_samples_list_expression=n_samples_list_expression,
        n_samples_list_histology=n_samples_list_histology,
        n_features_expression=N_GENES,
        n_features_histology=N_HISTOLOGY_CHANNELS,
        G_init_expression=torch.from_numpy(X_expression).float().clone(),
        G_init_histology=torch.from_numpy(X_histology).float().clone(),
    ).to(device)

    ## Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    x_expression = torch.from_numpy(X_expression).clone().float()
    y_expression = torch.from_numpy(Y_expression).clone().float()
    x_histology = torch.from_numpy(X_histology).clone().float()
    y_histology = torch.from_numpy(Y_histology).clone().float()

    def train(model, loss_fn, optimizer):
        model.train()

        # Forwrard pass
        (
            G_expression,
            G_histology,
            mean_G_expression,
            mean_G_histology,
            cov_G_list,
            mean_Y_expression,
            mean_Y_histology,
            cov_Y_expression,
            cov_Y_histology,
        ) = model.forward(x_expression, x_histology)

        # Compute loss
        loss = loss_fn(
            x_expression,
            x_histology,
            y_expression,
            y_histology,
            G_expression,
            G_histology,
            mean_G_expression,
            mean_G_histology,
            cov_G_list,
            mean_Y_expression,
            mean_Y_histology,
            cov_Y_expression,
            cov_Y_histology,
            view_idx_expression,
            view_idx_histology,
        )

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    data_expression_ax = fig.add_subplot(221, frameon=False)
    data_histology_ax = fig.add_subplot(222, frameon=False)
    latent_expression_ax = fig.add_subplot(223, frameon=False)
    latent_histology_ax = fig.add_subplot(224, frameon=False)

    plt.show(block=False)

    def callback(model):
        model.eval()
        markers = [".", "+", "^"]

        data_expression_ax.cla()
        data_histology_ax.cla()
        latent_expression_ax.cla()
        latent_histology_ax.cla()
        data_expression_ax.set_title("Expression data")
        data_histology_ax.set_title("Histology data")
        latent_expression_ax.set_title("G, Expression")
        latent_histology_ax.set_title("G, Histology")

        for vv in range(N_VIEWS):
            data_expression_ax.scatter(
                X_expression[view_idx_expression[vv], 0],
                X_expression[view_idx_expression[vv], 1],
                c=np.sum(Y_expression[view_idx_expression[vv], :], axis=1),
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            data_histology_ax.scatter(
                X_histology[view_idx_histology[vv], 0],
                X_histology[view_idx_histology[vv], 1],
                c=Y_histology[view_idx_histology[vv], :],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            latent_expression_ax.scatter(
                model.G_expression.detach().numpy()[view_idx_expression[vv], 0],
                model.G_expression.detach().numpy()[view_idx_expression[vv], 1],
                c=np.sum(Y_expression[view_idx_expression[vv], :], axis=1),
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            latent_histology_ax.scatter(
                model.G_histology.detach().numpy()[view_idx_histology[vv], 0],
                model.G_histology.detach().numpy()[view_idx_histology[vv], 1],
                c=Y_histology[view_idx_histology[vv], :],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
        plt.draw()
        plt.savefig("../plots/example_alignment_visium.png")
        plt.pause(1 / 60.0)

    loss_trace = []
    for t in range(N_EPOCHS):
        loss = train(model, loss_fn, optimizer)
        loss_trace.append(loss)
        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            callback(model)

    plt.close()

    # Set up figure.
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    data_expression_ax = fig.add_subplot(221, frameon=False)
    data_histology_ax = fig.add_subplot(222, frameon=False)
    latent_expression_ax = fig.add_subplot(223, frameon=False)
    latent_histology_ax = fig.add_subplot(224, frameon=False)

    model.eval()
    markers = [".", "+", "^"]

    data_expression_ax.cla()
    data_histology_ax.cla()
    latent_expression_ax.cla()
    latent_histology_ax.cla()
    data_expression_ax.set_title("Expression data")
    data_histology_ax.set_title("Histology data")
    latent_expression_ax.set_title("G, Expression")
    latent_histology_ax.set_title("G, Histology")

    for vv in range(N_VIEWS):
        data_expression_ax.scatter(
            X_expression[view_idx_expression[vv], 0],
            X_expression[view_idx_expression[vv], 1],
            c=np.sum(Y_expression[view_idx_expression[vv], :], axis=1),
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
        )
        data_expression_ax.set_xticks([])
        data_expression_ax.set_yticks([])
        data_histology_ax.scatter(
            X_histology[view_idx_histology[vv], 0],
            X_histology[view_idx_histology[vv], 1],
            c=Y_histology[view_idx_histology[vv], :],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
        )
        data_histology_ax.set_xticks([])
        data_histology_ax.set_yticks([])
        latent_expression_ax.scatter(
            model.G_expression.detach().numpy()[view_idx_expression[vv], 0],
            model.G_expression.detach().numpy()[view_idx_expression[vv], 1],
            c=np.sum(Y_expression[view_idx_expression[vv], :], axis=1),
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
        )
        latent_expression_ax.set_xticks([])
        latent_expression_ax.set_yticks([])
        latent_histology_ax.scatter(
            model.G_histology.detach().numpy()[view_idx_histology[vv], 0],
            model.G_histology.detach().numpy()[view_idx_histology[vv], 1],
            c=Y_histology[view_idx_histology[vv], :],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=100,
        )
        latent_histology_ax.set_xticks([])
        latent_histology_ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("../plots/example_alignment_visium.png")
    # plt.show()
    plt.close()

    import ipdb

    ipdb.set_trace()
