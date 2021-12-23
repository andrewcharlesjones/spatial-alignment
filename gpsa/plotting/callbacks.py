import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import pandas as pd
from scipy.stats import pearsonr

from matplotlib.lines import Line2D

import seaborn as sns

SCATTER_POINT_SIZE = 50


def callback_oned(
    model,
    X,
    Y,
    X_aligned,
    data_expression_ax,
    latent_expression_ax,
    prediction_ax=None,
    X_test=None,
    Y_pred=None,
    Y_test_true=None,
    X_test_aligned=None,
    F_samples=None,
):
    model.eval()
    markers = list(Line2D.markers.keys())
    colors = ["blue", "orange"]

    if model.fixed_view_idx is not None:
        curr_idx = model.view_idx["expression"][model.fixed_view_idx]
        X_aligned["expression"][curr_idx] = torch.tensor(X[curr_idx].astype(np.float32))

    data_expression_ax.cla()
    latent_expression_ax.cla()

    data_expression_ax.set_title("Observed data")
    latent_expression_ax.set_title("Aligned data")

    data_expression_ax.set_xlabel("Spatial coordinate")
    latent_expression_ax.set_xlabel("Spatial coordinate")

    data_expression_ax.set_ylabel("Outcome")
    latent_expression_ax.set_ylabel("Outcome")

    data_expression_ax.set_xlim([X.min(), X.max()])
    latent_expression_ax.set_xlim([X.min(), X.max()])

    for vv in range(model.n_views):

        view_idx = model.view_idx["expression"]

        data_expression_ax.scatter(
            X[view_idx[vv], 0],
            Y[view_idx[vv], 0],
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=SCATTER_POINT_SIZE,
            c="blue",
        )
        if Y.shape[1] > 1:
            data_expression_ax.scatter(
                X[view_idx[vv], 0],
                Y[view_idx[vv], 1],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=SCATTER_POINT_SIZE,
                c="orange",
            )
        latent_expression_ax.scatter(
            # model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
            X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
            Y[view_idx[vv], 0],
            c="blue",
            label="View {}".format(vv + 1),
            marker=markers[vv],
            s=SCATTER_POINT_SIZE,
        )
        if Y.shape[1] > 1:
            latent_expression_ax.scatter(
                # model.G_means["expression"].detach().numpy()[view_idx[vv], 0],
                X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
                Y[view_idx[vv], 1],
                c="orange",
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=SCATTER_POINT_SIZE,
            )
        # latent_expression_ax.scatter(
        # 	model.Xtilde.detach().numpy()[vv, :, 0],
        # 	model.delta_list.detach().numpy()[vv][:, 0],
        # 	c="red",
        # 	label="View {}".format(vv + 1),
        # 	marker="^",
        # 	s=100,
        # )

        if F_samples is not None:
            latent_expression_ax.scatter(
                X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
                F_samples.detach().numpy()[view_idx[vv], 0],
                c="red",
                marker=markers[vv],
                s=SCATTER_POINT_SIZE,
            )
            if Y.shape[1] > 1:
                latent_expression_ax.scatter(
                    X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
                    F_samples.detach().numpy()[view_idx[vv], 1],
                    c="green",
                    marker=markers[vv],
                    s=SCATTER_POINT_SIZE,
                )

    if prediction_ax is not None:

        prediction_ax.cla()
        prediction_ax.set_title("Predictions")
        prediction_ax.set_xlabel("True outcome")
        prediction_ax.set_ylabel("Predicted outcome")

        ### Plots the warping function
        # prediction_ax.scatter(
        # 	X[view_idx[vv], 0],
        # 	X_aligned["expression"].detach().numpy()[view_idx[vv], 0],
        # 	label="View {}".format(vv + 1),
        # 	marker=markers[vv],
        # 	s=100,
        # 	c="blue",
        # )
        # prediction_ax.scatter(
        # 	model.Xtilde.detach().numpy()[vv, :, 0],
        # 	model.delta_list.detach().numpy()[vv][:, 0],
        # 	c="red",
        # 	label="View {}".format(vv + 1),
        # 	marker="^",
        # 	s=100,
        # )
        latent_expression_ax.scatter(
            X_test_aligned["expression"].detach().numpy()[:, 0],
            Y_pred.detach().numpy()[:, 0],
            c="blue",
            label="Prediction",
            marker="^",
            s=SCATTER_POINT_SIZE,
        )
        latent_expression_ax.scatter(
            X_test_aligned["expression"].detach().numpy()[:, 0],
            Y_pred.detach().numpy()[:, 1],
            c="orange",
            label="Prediction",
            marker="^",
            s=SCATTER_POINT_SIZE,
        )
        prediction_ax.scatter(
            Y_test_true[:, 0],
            Y_pred.detach().numpy()[:, 0],
            c="black",
            s=SCATTER_POINT_SIZE,
        )
        prediction_ax.scatter(
            Y_test_true[:, 1],
            Y_pred.detach().numpy()[:, 1],
            c="black",
            s=SCATTER_POINT_SIZE,
            marker="^",
        )

    data_expression_ax.legend()
    plt.draw()
    plt.pause(1 / 60.0)


def callback_twod(
    model,
    X,
    Y,
    X_aligned,
    data_expression_ax,
    latent_expression_ax,
    is_mle=False,
    gene_idx=0,
    s=200,
):

    if model.fixed_view_idx is not None:
        if is_mle:
            pass
        else:
            curr_idx = model.view_idx["expression"][model.fixed_view_idx]
            X_aligned["expression"][curr_idx] = torch.tensor(
                X[curr_idx].astype(np.float32)
            )

    model.eval()
    markers = [".", "+", "^"]
    colors = ["blue", "orange"]

    data_expression_ax.cla()
    latent_expression_ax.cla()
    data_expression_ax.set_title("Observed data")
    latent_expression_ax.set_title("Aligned data")

    curr_view_idx = model.view_idx["expression"]

    latent_Xs = []
    Xs = []
    Ys = []
    markers_list = []
    viewname_list = []

    for vv in range(model.n_views):

        ## Data
        Xs.append(X[curr_view_idx[vv]])

        ## Latents
        curr_latent_Xs = X_aligned["expression"].detach().numpy()[curr_view_idx[vv]]
        latent_Xs.append(curr_latent_Xs)
        Ys.append(Y[curr_view_idx[vv], gene_idx])
        markers_list.append([markers[vv]] * curr_latent_Xs.shape[0])
        viewname_list.append(["View {}".format(vv + 1)] * curr_latent_Xs.shape[0])

    Xs = np.concatenate(Xs, axis=0)
    latent_Xs = np.concatenate(latent_Xs, axis=0)
    Ys = np.concatenate(Ys)
    markers_list = np.concatenate(markers_list)
    viewname_list = np.concatenate(viewname_list)

    data_df = pd.DataFrame(
        {
            "X1": Xs[:, 0],
            "X2": Xs[:, 1],
            "Y": Ys,
            "marker": markers_list,
            "view": viewname_list,
        }
    )

    latent_df = pd.DataFrame(
        {
            "X1": latent_Xs[:, 0],
            "X2": latent_Xs[:, 1],
            "Y": Ys,
            "marker": markers_list,
            "view": viewname_list,
        }
    )

    plt.sca(data_expression_ax)
    g = sns.scatterplot(
        data=data_df,
        x="X1",
        y="X2",
        hue="Y",
        style="view",
        ax=data_expression_ax,
        s=s,
        linewidth=1.8,
        edgecolor="black",
        palette="viridis",
    )
    g.legend_.remove()
    # plt.colorbar()
    # plt.axis("off")
    # plt.scatter(model.Xtilde.detach().numpy()[0, :, 0], model.Xtilde.detach().numpy()[0, :, 1], color="red")
    # plt.scatter(model.Xtilde.detach().numpy()[1, :, 0], model.Xtilde.detach().numpy()[1, :, 1], color="red")
    # plt.scatter(model.Gtilde.detach().numpy()[:, 0], model.Gtilde.detach().numpy()[:, 1], color="red")
    # plt.axis("off")

    plt.sca(latent_expression_ax)
    g = sns.scatterplot(
        data=latent_df,
        x="X1",
        y="X2",
        hue="Y",
        style="view",
        ax=latent_expression_ax,
        s=s,
        linewidth=1.8,
        edgecolor="black",
        palette="viridis",
    )
    g.legend_.remove()
    # plt.colorbar()

    # import ipdb; ipdb.set_trace()

    # for vv in range(model.n_views):

    #     # import ipdb; ipdb.set_trace()
    #     data_expression_ax.scatter(
    #         X[curr_view_idx[vv], 0],
    #         X[curr_view_idx[vv], 1],
    #         c=Y[curr_view_idx[vv], 0],
    #         label="View {}".format(vv + 1),
    #         marker=markers[vv],
    #         s=400,
    #     )
    # latent_expression_ax.scatter(
    #     X_aligned["expression"].detach().numpy()[curr_view_idx[vv], 0],
    #     X_aligned["expression"].detach().numpy()[curr_view_idx[vv], 1],
    #     c=Y[curr_view_idx[vv], 0],
    #     label="View {}".format(vv + 1),
    #     marker=markers[vv],
    #     s=400,
    # )
    # plt.axis("off")


def callback_twod_aligned_only(
    model,
    X,
    Y,
    X_aligned,
    latent_expression_ax1,
    latent_expression_ax2,
    is_mle=False,
    gene_idx=0,
):

    if model.fixed_view_idx is not None:
        if is_mle:
            pass
        else:
            curr_idx = model.view_idx["expression"][model.fixed_view_idx]
            X_aligned["expression"][curr_idx] = torch.tensor(
                X[curr_idx].astype(np.float32)
            )

    model.eval()
    markers = [".", "+", "^"]
    colors = ["blue", "orange"]

    latent_expression_ax1.cla()
    latent_expression_ax2.cla()
    latent_expression_ax1.set_title("Observed data")
    latent_expression_ax2.set_title("Aligned data")

    curr_view_idx = model.view_idx["expression"]

    latent_Xs = []
    Xs = []
    Ys = []
    markers_list = []
    viewname_list = []

    aligned_coords = X_aligned["expression"].detach().numpy()

    for vv in range(model.n_views):

        ## Data
        Xs.append(X[curr_view_idx[vv]])

        ## Latents
        curr_latent_Xs = aligned_coords[curr_view_idx[vv]]
        latent_Xs.append(curr_latent_Xs)
        Ys.append(Y[curr_view_idx[vv], gene_idx])
        markers_list.append([markers[vv]] * curr_latent_Xs.shape[0])
        viewname_list.append(["View {}".format(vv + 1)] * curr_latent_Xs.shape[0])

    latent_expression_ax1.scatter(
        aligned_coords[curr_view_idx[0]][:, 0],
        aligned_coords[curr_view_idx[0]][:, 1],
        c=Y[curr_view_idx[0]][:, gene_idx].squeeze(),
        s=24,
        marker="h",
    )
    latent_expression_ax2.scatter(
        aligned_coords[curr_view_idx[1]][:, 0],
        aligned_coords[curr_view_idx[1]][:, 1],
        c=Y[curr_view_idx[1]][:, gene_idx].squeeze(),
        s=24,
        marker="h",
    )
    # latent_expression_ax1.scatter(model.Xtilde.detach().numpy()[0, :, 0], model.Xtilde.detach().numpy()[0, :, 1], color="red")
    # latent_expression_ax2.scatter(model.Xtilde.detach().numpy()[1, :, 0], model.Xtilde.detach().numpy()[1, :, 1], color="red")

    plt.axis("off")


def callback_twod_multimodal(
    model, data_dict, X_aligned, axes, rgb=False, scatterpoint_size=100
):

    # if model.fixed_view_idx is not None:
    #     if is_mle:
    #         pass
    #     else:
    #         curr_idx = model.view_idx["expression"][model.fixed_view_idx]
    #         X_aligned["expression"][curr_idx] = torch.tensor(X[curr_idx].astype(np.float32))

    model.eval()
    markers = [".", "+", "^"]
    colors = ["blue", "orange"]

    [ax.cla() for ax in axes]

    axes[0].set_title("Observed expression")
    axes[1].set_title("Aligned expression")
    axes[2].set_title("Observed histology")
    axes[3].set_title("Aligned histology")

    axis_counter = 0
    n_mods = 2
    for mod in ["expression", "histology"]:
        curr_view_idx = model.view_idx[mod]
        for vv in range(model.n_views):

            # import ipdb; ipdb.set_trace()
            curr_coords = data_dict[mod]["spatial_coords"]

            if mod == "histology" and rgb:
                curr_outputs = data_dict[mod]["outputs"][curr_view_idx[vv], :]
            else:
                curr_outputs = data_dict[mod]["outputs"][curr_view_idx[vv], 0]
            axes[axis_counter].scatter(
                curr_coords[curr_view_idx[vv], 0],
                curr_coords[curr_view_idx[vv], 1],
                c=curr_outputs,
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=scatterpoint_size,
            )
            axes[axis_counter + 1].scatter(
                X_aligned[mod].detach().numpy()[curr_view_idx[vv], 0],
                X_aligned[mod].detach().numpy()[curr_view_idx[vv], 1],
                c=curr_outputs,
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=scatterpoint_size,
            )
        axis_counter += n_mods
