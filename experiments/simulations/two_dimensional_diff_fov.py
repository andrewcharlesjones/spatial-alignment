import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys

sys.path.append("../..")
from models.gpsa_vi_lmc import VariationalWarpGP

sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data_partial_overlap
from plotting.callbacks import callback_twod
from util import ConvergenceChecker

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../paste")
from src.paste import PASTE, visualization

LATEX_FONTSIZE = 30
import matplotlib

font = {"size": LATEX_FONTSIZE}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


device = "cuda" if torch.cuda.is_available() else "cpu"


n_spatial_dims = 2
n_views = 2
# n_outputs = 10
m_G = 40
m_X_per_view = 40

N_EPOCHS = 1000
PRINT_EVERY = 50


def two_d_gpsa_diff_fov(
    n_outputs,
    n_epochs,
    n_latent_gps,
    warp_kernel_variance=0.1,
    plot_intermediate=True,
    make_plot=False,
):

    X, Y, n_samples_list, view_idx, keep_idx = generate_twod_data_partial_overlap(
        n_views,
        n_outputs,
        grid_size=15,
        n_latent_gps=n_latent_gps["expression"],
        kernel_lengthscale=10.0,
        kernel_variance=warp_kernel_variance,
    )

    X = X - X.min(0)
    X = X / X.max(0) * 10

    ##  PASTE
    slice1 = anndata.AnnData(np.exp(Y[view_idx[0]]))
    slice2 = anndata.AnnData(np.exp(Y[view_idx[1]]))

    slice1.obsm["spatial"] = X[view_idx[0]]
    slice2.obsm["spatial"] = X[view_idx[1]]

    pi12 = PASTE.pairwise_align(slice1, slice2, alpha=0.1)

    slices = [slice1, slice2]
    pis = [pi12]
    new_slices = visualization.stack_slices_pairwise(slices, pis)

    err_paste = np.mean(
        np.sum(
            (new_slices[0].obsm["spatial"][keep_idx] - new_slices[1].obsm["spatial"])
            ** 2,
            axis=1,
        )
    )

    # visualization.plot_slice(new_slices[0], color="red")
    # visualization.plot_slice(new_slices[1], color="blue")
    # plt.show()

    # import ipdb; ipdb.set_trace()

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()

    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }

    model = VariationalWarpGP(
        data_dict,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=False,
        minmax_init=False,
        grid_init=True,
        n_latent_gps=n_latent_gps,
        # n_latent_gps=None,
        mean_function="identity_fixed",
        fixed_warp_kernel_variances=np.ones(n_views) * 0.1,
        fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        # fixed_data_kernel_lengthscales=np.exp(gpr.kernel_.k1.theta.astype(np.float32)),
        # fixed_data_kernel_lengthscales=np.exp(gpr.kernel_.k1.theta[0]),
        # mean_function="identity_initialized",
        # fixed_view_idx=0,
    ).to(device)

    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            {"expression": x}, view_idx=view_idx, Ns=Ns, S=1
        )

        # Compute loss
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(14, 7), facecolor="white", constrained_layout=True)
    data_expression_ax = fig.add_subplot(122, frameon=False)
    latent_expression_ax = fig.add_subplot(121, frameon=False)
    plt.show(block=False)

    convergence_checker = ConvergenceChecker(span=100)

    loss_trace = []
    error_trace = []
    for t in range(n_epochs):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)

        # if plot_intermediate and t % PRINT_EVERY == 0:
        #     print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
        #     G_means, G_samples, F_latent_samples, F_samples = model.forward(
        #         {"expression": x}, view_idx=view_idx, Ns=Ns
        #     )
        #     callback_twod(
        #         model,
        #         X,
        #         Y,
        #         data_expression_ax=data_expression_ax,
        #         latent_expression_ax=latent_expression_ax,
        #         X_aligned=G_means,
        #     )
        #     plt.draw()
        #     plt.pause(1 / 60.0)

        #     curr_X_aligned = G_means["expression"].detach().numpy()
        #     err = np.mean(np.sum((curr_X_aligned[view_idx["expression"][0][keep_idx]] - curr_X_aligned[view_idx["expression"][1]])**2, axis=1))
        #     print("Error: {}".format(err))
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        {"expression": x}, view_idx=view_idx, Ns=Ns
    )

    curr_X_aligned = G_means["expression"].detach().numpy()
    err_gpsa = np.mean(
        np.sum(
            (
                curr_X_aligned[view_idx["expression"][0][keep_idx]]
                - curr_X_aligned[view_idx["expression"][1]]
            )
            ** 2,
            axis=1,
        )
    )
    print("Error: {}".format(err_gpsa))

    if make_plot:
        plt.close()
        plt.figure(figsize=(21, 7))

        viewname_list = []
        markers_list = []
        for vv in range(n_views):
            viewname_list.append(
                ["View {}".format(vv + 1)] * view_idx["expression"][vv].shape[0]
            )

        viewname_list = np.concatenate(viewname_list)

        SCATTERPOINT_SIZE = 200

        plt.subplot(131)
        plot_df = pd.DataFrame(
            {
                "X1": X[:, 0],
                "X2": X[:, 1],
                "Y": Y[:, 0],
                "view": viewname_list,
            }
        )
        g = sns.scatterplot(
            data=plot_df,
            x="X1",
            y="X2",
            hue="Y",
            style="view",
            s=SCATTERPOINT_SIZE,
            linewidth=0.4,
            palette="viridis",
        )
        g.legend_.remove()
        plt.axis("off")
        plt.title("Data")

        plt.subplot(132)
        paste_aligned_coords = np.concatenate(
            [new_slices[0].obsm["spatial"], new_slices[1].obsm["spatial"]], axis=0
        )
        plot_df = pd.DataFrame(
            {
                "X1": paste_aligned_coords[:, 0],
                "X2": paste_aligned_coords[:, 1],
                "Y": Y[:, 0],
                "view": viewname_list,
            }
        )
        g = sns.scatterplot(
            data=plot_df,
            x="X1",
            y="X2",
            hue="Y",
            style="view",
            s=SCATTERPOINT_SIZE,
            linewidth=0.4,
            palette="viridis",
        )
        g.legend_.remove()
        plt.axis("off")
        plt.title("PASTE")

        plt.subplot(133)
        plot_df = pd.DataFrame(
            {
                "X1": G_means["expression"].detach().numpy()[:, 0],
                "X2": G_means["expression"].detach().numpy()[:, 1],
                "Y": Y[:, 0],
                "view": viewname_list,
            }
        )
        g = sns.scatterplot(
            data=plot_df,
            x="X1",
            y="X2",
            hue="Y",
            style="view",
            s=SCATTERPOINT_SIZE,
            linewidth=0.4,
            palette="viridis",
        )
        g.legend_.remove()
        plt.axis("off")
        plt.title("GPSA")
        plt.savefig("./out/partial_overlap_comparison_alignments.png")
        plt.show()
        plt.close()

    plt.close()

    return X, Y, G_means, model, err_paste, err_gpsa


if __name__ == "__main__":

    n_outputs = 10
    n_repeats = 5

    errs_paste = np.zeros(n_repeats)
    errs_gpsa = np.zeros(n_repeats)

    for ii in range(n_repeats):
        X, Y, G_means, model, err_paste, err_gpsa = two_d_gpsa_diff_fov(
            n_epochs=N_EPOCHS,
            n_outputs=n_outputs,
            warp_kernel_variance=0.2,
            n_latent_gps={"expression": 3},
            make_plot=True if ii == n_repeats - 1 else False,
        )

        errs_paste[ii] = err_paste
        errs_gpsa[ii] = err_gpsa

    err_df = pd.DataFrame({"PASTE": errs_paste, "GPSA": errs_gpsa})
    err_df = pd.melt(err_df)

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=err_df, x="variable", y="value", color="gray")
    plt.xlabel("")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig("./out/partial_overlap_comparison.png")
    plt.close()
    import ipdb

    ipdb.set_trace()
