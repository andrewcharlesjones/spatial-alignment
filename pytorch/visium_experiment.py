import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy
import pandas as pd

# from warp_gp_multimodal import TwoLayerWarpGP
import sys
from variational_warp_gp_lmc2 import VariationalWarpGP

sys.path.append("..")
from util import get_st_coordinates, polar_warp
import torch
from torch.utils.data import Dataset, DataLoader
import socket
from os.path import join as pjoin
from sklearn.gaussian_process.kernels import RBF
import pyro.contrib.gp as gp
from gp_functions import rbf_covariance

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

N_VIEWS = 2
N_SAMPLES = 300
N_GENES = 30
N_HISTOLOGY_CHANNELS = 3
SCALEFACTOR = 0.17211704
GRAY_PIXEL_VAL = 0.7
WARP_SCALE = 0.1
PRINT_EVERY = 25
LEARNING_RATE = 1e-2
N_EPOCHS = 4
DISTANCE_PENALTY_PARAM = 1e-2
MEAN_PENALTY_PARAM = 1e0
N_LATENT_GPS = 2

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
    # import ipdb; ipdb.set_trace()

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

    assert X_orig_sample1.shape[0] == Y_orig_unstdized_sample1.shape[0]
    assert X_orig_sample2.shape[0] == Y_orig_unstdized_sample2.shape[0]

    ## Subsample expression locations
    n1_expression = X1_expression.shape[0]
    expression1_subsample_idx = np.random.choice(
        np.arange(n1_expression), size=N_SAMPLES, replace=False
    )
    X1_expression = X1_expression[expression1_subsample_idx]
    X2_expression = X2_expression[expression1_subsample_idx]
    Y1_expression = Y_orig_unstdized_sample1[expression1_subsample_idx]
    Y2_expression = Y_orig_unstdized_sample2[expression1_subsample_idx]

    ## Standardize expression
    Y1_expression = (Y1_expression - Y1_expression.mean(0)) / Y1_expression.std(0)
    Y2_expression = (Y2_expression - Y2_expression.mean(0)) / Y2_expression.std(0)

    # plt.hist(Y1_expression[:, 0])
    # plt.show()
    # import ipdb; ipdb.set_trace()

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
    X1 = np.concatenate([X1_expression, X1_histology]) / max_val * 20
    X2 = np.concatenate([X2_expression, X2_histology]) / max_val * 20

    ## Warp coordinates
    # linear_coeffs = np.random.normal(scale=WARP_SCALE, size=N_VIEWS * 2 * 2)
    # # linear_coeffs = np.concatenate([[-0.2] * 4, [0.2] * 4])
    # r1s_true, theta1s_true = X1 @ linear_coeffs[:2], X1 @ linear_coeffs[2:4]
    # r2s_true, theta2s_true = X2 @ linear_coeffs[4:6], X2 @ linear_coeffs[6:]

    # X1_observed = polar_warp(X1, r1s_true, theta1s_true)
    # X2_observed = polar_warp(X2, r2s_true, theta2s_true)
    X1_observed = X1 @ (
        np.eye(2) + np.random.normal(0, 0.05, size=(2, 2))
    )  # + np.random.normal(scale=0.25, size=X_orig_single.shape)
    X2_observed = X2 @ (
        np.eye(2) + np.random.normal(0, 0.05, size=(2, 2))
    )  # + np.random.normal(scale=0.25, size=X_orig_single.shape)

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

    ####

    # xlimits = [-10, 10]
    # ylimits = [-10, 10]
    # numticks = 15
    # x1s = np.linspace(*xlimits, num=numticks)
    # x2s = np.linspace(*ylimits, num=numticks)
    # X1, X2 = np.meshgrid(x1s, x2s)
    # X_orig_single = np.vstack([X1.ravel(), X2.ravel()]).T
    # # X_expression = np.concatenate([X_orig_single.copy(), X_orig_single.copy()], axis=0)
    # X1_observed = X_orig_single @ (np.eye(2) + np.random.normal(0, 0.05, size=(2, 2))) #+ np.random.normal(scale=0.25, size=X_orig_single.shape)
    # X2_observed = X_orig_single @ (np.eye(2) + np.random.normal(0, 0.05, size=(2, 2))) #+ np.random.normal(scale=0.25, size=X_orig_single.shape)
    # X_expression = np.concatenate([X1_observed.copy(), X2_observed.copy()], axis=0)

    # N_LATENT_GPS = 2
    # kernel = rbf_covariance
    # kernel_params_true = np.array([np.log(1.0), np.log(1.0)])
    # K_XX = kernel(X_orig_single, X_orig_single, kernel_params_true)
    # Y_orig_latent = np.vstack(
    # 	[
    # 		mvnpy.rvs(
    # 			mean=np.zeros(X_orig_single.shape[0]),
    # 			cov=K_XX,
    # 		)
    # 		for _ in range(N_LATENT_GPS)
    # 	]
    # ).T
    # W_mat = np.random.normal(size=(N_LATENT_GPS, N_GENES))
    # Y_orig = Y_orig_latent @ W_mat
    # Y_expression = np.concatenate([Y_orig, Y_orig], axis=0)
    # n_samples_list_expression = [Y_orig.shape[0]] * 2
    # view_idx_expression = np.array(
    # 	[
    # 		np.arange(0, Y_orig.shape[0]),
    # 		np.arange(Y_orig.shape[0], Y_orig.shape[0] * 2),
    # 	]
    # )
    # import ipdb; ipdb.set_trace()

    x_expression = torch.from_numpy(X_expression).clone().float()
    y_expression = torch.from_numpy(Y_expression).clone().float()
    x_histology = torch.from_numpy(X_histology).clone().float()
    y_histology = torch.from_numpy(Y_histology).clone().float()

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

    ## Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train(model, loss_fn, optimizer):
        model.train()

        # import ipdb; ipdb.set_trace()
        # Forwrard pass
        G_samples, F_samples = model.forward(
            {"expression": x_expression, "histology": x_histology}
        )

        # Compute loss
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # import ipdb; ipdb.set_trace()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(20, 5), facecolor="white")
    data_expression_ax = fig.add_subplot(141, frameon=False)
    data_histology_ax = fig.add_subplot(142, frameon=False)
    latent_expression_ax = fig.add_subplot(143, frameon=False)
    latent_histology_ax = fig.add_subplot(144, frameon=False)

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
                model.G_means["expression"]
                .detach()
                .numpy()[view_idx_expression[vv], 0],
                model.G_means["expression"]
                .detach()
                .numpy()[view_idx_expression[vv], 1],
                c=np.sum(Y_expression[view_idx_expression[vv], :], axis=1),
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            latent_expression_ax.scatter(
                model.Xtilde[vv].detach().numpy()[:, 0],
                model.Xtilde[vv].detach().numpy()[:, 1],
                c="red",
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            latent_histology_ax.scatter(
                model.G_means["histology"].detach().numpy()[view_idx_histology[vv], 0],
                model.G_means["histology"].detach().numpy()[view_idx_histology[vv], 1],
                c=Y_histology[view_idx_histology[vv], :],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            latent_histology_ax.scatter(
                model.Xtilde[vv].detach().numpy()[:, 0],
                model.Xtilde[vv].detach().numpy()[:, 1],
                c="red",
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
        plt.draw()
        # if model.distance_penalty_param > 0:
        #     plt.savefig("../plots/example_alignment_visium.png")
        # else:
        #     plt.savefig("../plots/example_alignment_visium_unconstrained.png")
        plt.pause(1 / 60.0)

    loss_trace = []
    for t in range(N_EPOCHS):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)
        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            callback(model)

    plt.close()

    # Set up figure.
    fig = plt.figure(figsize=(28, 7), facecolor="white")
    data_expression_ax = fig.add_subplot(141, frameon=False)
    data_histology_ax = fig.add_subplot(142, frameon=False)
    latent_expression_ax = fig.add_subplot(143, frameon=False)
    latent_histology_ax = fig.add_subplot(144, frameon=False)

    model.eval()
    markers = [".", "+", "^"]

    data_expression_ax.cla()
    data_histology_ax.cla()
    latent_expression_ax.cla()
    latent_histology_ax.cla()
    data_expression_ax.set_title("Expression data")
    data_histology_ax.set_title("Histology data")
    latent_expression_ax.set_title("Expression, aligned")
    latent_histology_ax.set_title("Histology, aligned")
    # data_expression_ax.set_xlabel("Spatial coordinate 1")
    # data_expression_ax.set_ylabel("Spatial coordinate 2")

    for vv in range(N_VIEWS):
        data_expression_ax.scatter(
            X_expression[view_idx_expression[vv], 0],
            X_expression[view_idx_expression[vv], 1],
            c=np.sum(Y_expression[view_idx_expression[vv], :], axis=1),
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=200,
        )
        data_expression_ax.set_xticks([])
        data_expression_ax.set_yticks([])
        data_histology_ax.scatter(
            X_histology[view_idx_histology[vv], 0],
            X_histology[view_idx_histology[vv], 1],
            c=Y_histology[view_idx_histology[vv], :],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=200,
        )
        data_histology_ax.set_xticks([])
        data_histology_ax.set_yticks([])
        latent_expression_ax.scatter(
            model.G_means["expression"].detach().numpy()[view_idx_expression[vv], 0],
            model.G_means["expression"].detach().numpy()[view_idx_expression[vv], 1],
            c=np.sum(Y_expression[view_idx_expression[vv], :], axis=1),
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=200,
        )
        latent_expression_ax.set_xticks([])
        latent_expression_ax.set_yticks([])
        latent_histology_ax.scatter(
            model.G_means["histology"].detach().numpy()[view_idx_histology[vv], 0],
            model.G_means["histology"].detach().numpy()[view_idx_histology[vv], 1],
            c=Y_histology[view_idx_histology[vv], :],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=200,
        )
        latent_histology_ax.set_xticks([])
        latent_histology_ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("../plots/example_alignment_visium.png")
    # plt.show()
    plt.close()

    import ipdb

    ipdb.set_trace()
