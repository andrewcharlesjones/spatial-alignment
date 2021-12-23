import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import sys

sys.path.append("../..")
from models.gpsa_vi_lmc import VariationalWarpGP
from util import matern12_kernel, rbf_kernel


sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data
from plotting.callbacks import callback_twod
from util import ConvergenceChecker

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../paste")
from src.paste import PASTE, visualization


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 35

n_spatial_dims = 2
n_views = 2
# n_outputs = 10
m_G = 50
m_X_per_view = 50

N_EPOCHS = 2000
PRINT_EVERY = 1500
# N_LATENT_GPS = 1


warp_kernel_variance = 0.5
noise_variance = 0.001
n_latent_gps = {"expression": 5}
n_outputs = 30

# import ipdb; ipdb.set_trace()


def two_d_gpsa(
    X,
    Y,
    n_samples_list,
    view_idx,
    n_outputs,
    n_epochs,
    n_latent_gps,
    warp_kernel_variance=0.1,
    noise_variance=0.0,
    plot_intermediate=True,
    fixed_view_idx=None,
):

    n_samples_per_view = X.shape[0] // n_views

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
            (new_slices[0].obsm["spatial"] - new_slices[1].obsm["spatial"]) ** 2, axis=1
        )
    )

    ## Fit GP on one view to get initial estimates of data kernel parameters
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.gaussian_process import GaussianProcessRegressor

    kernel = RBF(length_scale=1.0) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X[view_idx[0]], Y[view_idx[0]])
    data_lengthscales_est = gpr.kernel_.k1.theta[0]
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
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=n_latent_gps,
        # n_latent_gps=None,
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
        fixed_warp_kernel_variances=np.ones(n_views) * 1.0,
        # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        fixed_view_idx=fixed_view_idx,
    ).to(device)

    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            {"expression": x}, view_idx=view_idx, Ns=Ns, S=5
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

    loss_trace = []
    error_trace = []
    for t in range(n_epochs):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)

        if plot_intermediate and t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            G_means, G_samples, F_latent_samples, F_samples = model.forward(
                {"expression": x}, view_idx=view_idx, Ns=Ns
            )

            callback_twod(
                model,
                X,
                Y,
                data_expression_ax=data_expression_ax,
                latent_expression_ax=latent_expression_ax,
                X_aligned=G_means,
                s=600,
            )
            plt.draw()
            plt.pause(1 / 60.0)

        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            {"expression": x}, view_idx=view_idx, Ns=Ns
        )

    print("Done!")

    plt.close()

    return X, Y, G_means, model, err_paste


if __name__ == "__main__":

    X, Y, n_samples_list, view_idx = generate_twod_data(
        n_views,
        n_outputs,
        grid_size=10,
        n_latent_gps=n_latent_gps["expression"],
        kernel_lengthscale=5.0,
        kernel_variance=warp_kernel_variance,
        noise_variance=noise_variance,
        fixed_view_idx=0,
    )

    X, Y, G_means_denovo, model_denovo, err_paste = two_d_gpsa(
        X,
        Y,
        n_samples_list,
        view_idx,
        n_epochs=N_EPOCHS,
        n_outputs=n_outputs,
        warp_kernel_variance=0.5,
        noise_variance=0.001,
        n_latent_gps={"expression": 5},
    )

    X, Y, G_means_template, model_template, err_paste = two_d_gpsa(
        X,
        Y,
        n_samples_list,
        view_idx,
        n_epochs=N_EPOCHS,
        n_outputs=n_outputs,
        warp_kernel_variance=0.5,
        noise_variance=0.001,
        n_latent_gps={"expression": 5},
        fixed_view_idx=0,
    )

    import matplotlib

    font = {"size": LATEX_FONTSIZE}
    matplotlib.rc("font", **font)
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["xtick.labelsize"] = LATEX_FONTSIZE // 2
    matplotlib.rcParams["ytick.labelsize"] = LATEX_FONTSIZE // 2

    fig = plt.figure(figsize=(23, 7))
    data_ax = fig.add_subplot(131, frameon=False)
    denovo_ax = fig.add_subplot(132, frameon=False)
    template_ax = fig.add_subplot(133, frameon=False)

    data_ax.set_xlabel("Spatial 1")
    data_ax.set_ylabel("Spatial 2")
    denovo_ax.set_xlabel("Spatial 1")
    denovo_ax.set_ylabel("Spatial 2")
    template_ax.set_xlabel("Spatial 1")
    template_ax.set_ylabel("Spatial 2")

    data_ax.set_title("Data")
    denovo_ax.set_title(r"$\emph{De novo}$ alignment")
    template_ax.set_title(r"$\emph{Template-based}$ alignment")

    markers = ["o", "X"]
    edgecolors = ["black", "gray"]
    linewidth = 3
    for vv in range(model_denovo.n_views):
        curr_idx = model_denovo.view_idx["expression"][vv]
        data_ax.scatter(
            X[curr_idx][:, 0],
            X[curr_idx][:, 1],
            c=Y[curr_idx][:, 0],
            marker=markers[vv],
            s=500,
            edgecolor=edgecolors[vv],
            linewidth=linewidth,
        )

        curr_aligned_coords_denovo = (
            G_means_denovo["expression"].detach().numpy()[curr_idx]
        )
        denovo_ax.scatter(
            curr_aligned_coords_denovo[:, 0],
            curr_aligned_coords_denovo[:, 1],
            c=Y[curr_idx][:, 0],
            marker=markers[vv],
            s=500,
            edgecolor=edgecolors[vv],
            linewidth=linewidth,
        )

        curr_aligned_coords_template = (
            G_means_template["expression"].detach().numpy()[curr_idx]
        )
        if vv == 0:
            template_ax.scatter(
                X[curr_idx][:, 0],
                X[curr_idx][:, 1],
                c=Y[curr_idx][:, 0],
                marker=markers[vv],
                s=500,
                edgecolor=edgecolors[vv],
                linewidth=linewidth,
                label="View {}".format(vv + 1),
            )
        else:
            template_ax.scatter(
                curr_aligned_coords_template[:, 0],
                curr_aligned_coords_template[:, 1],
                c=Y[curr_idx][:, 0],
                marker=markers[vv],
                s=500,
                edgecolor=edgecolors[vv],
                linewidth=linewidth,
                label="View {}".format(vv + 1),
            )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig("../../plots/two_d_toy_example.png")
    plt.show()

    import ipdb

    ipdb.set_trace()
