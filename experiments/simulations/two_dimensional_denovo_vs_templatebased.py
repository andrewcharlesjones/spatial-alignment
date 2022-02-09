import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import sys
from gpsa import VariationalGPSA, matern12_kernel, rbf_kernel, LossNotDecreasingChecker
from gpsa.plotting import callback_twod

sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data

import matplotlib.animation as animation
import matplotlib.image as mpimg
import os
from os.path import join as pjoin
import anndata

import matplotlib

font = {"size": 25}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 35

n_spatial_dims = 2
n_views = 2
m_G = 50
m_X_per_view = 50

N_EPOCHS = 3000
PRINT_EVERY = 100


def two_d_gpsa(
    n_outputs,
    n_epochs,
    n_latent_gps,
    warp_kernel_variance=0.1,
    noise_variance=0.0,
    plot_intermediate=True,
    fixed_view_data=None,
    fixed_view_idx=None,
):

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()

    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }

    model = VariationalGPSA(
        data_dict,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=n_latent_gps,
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
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

    loss_trace = []
    error_trace = []
    convergence_checker = LossNotDecreasingChecker(max_epochs=n_epochs, atol=1e-4)

    for t in range(n_epochs):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)

        # has_converged = convergence_checker.check_loss(t, loss_trace)
        # if has_converged:
        #     print("Convergence criterion met.")
        #     break

        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))

        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            {"expression": x}, view_idx=view_idx, Ns=Ns
        )

    print("Done!")
    return G_means["expression"].detach().numpy()


if __name__ == "__main__":

    ## Generate data
    n_outputs = 30
    n_latent_gps = {"expression": 5}
    warp_kernel_variance = 0.5
    noise_variance = 0.001
    fixed_view_data = 0
    X, Y, n_samples_list, view_idx = generate_twod_data(
        n_views,
        n_outputs,
        grid_size=10,
        n_latent_gps=n_latent_gps["expression"],
        kernel_lengthscale=5.0,
        kernel_variance=warp_kernel_variance,
        noise_variance=noise_variance,
        fixed_view_idx=fixed_view_data,
    )
    n_samples_per_view = X.shape[0] // n_views

    ## Set up figure
    plt.figure(figsize=(18, 5))

    ## Plot data
    markers = ["o", "X"]
    plt.subplot(131)
    for vv in range(n_views):
        plt.scatter(
            X[view_idx[vv], 0],
            X[view_idx[vv], 1],
            c=Y[view_idx[vv], 0],
            marker=markers[vv],
            s=300,
            linewidth=1.8,
            edgecolor="black",
        )
    plt.title("Data")
    plt.xlabel("Spatial 1")
    plt.ylabel("Spatial 2")

    ## De novo
    aligned_coords_denovo = two_d_gpsa(
        n_epochs=N_EPOCHS,
        n_outputs=n_outputs,
        warp_kernel_variance=warp_kernel_variance,
        noise_variance=noise_variance,
        n_latent_gps=n_latent_gps,
        fixed_view_idx=None,
    )
    plt.subplot(132)
    for vv in range(n_views):
        plt.scatter(
            aligned_coords_denovo[view_idx[vv], 0],
            aligned_coords_denovo[view_idx[vv], 1],
            c=Y[view_idx[vv], 0],
            marker=markers[vv],
            s=300,
            linewidth=1.8,
            edgecolor="black",
        )
    plt.title(r"$\emph{De novo}$ alignment")
    plt.xlabel("Spatial 1")
    plt.ylabel("Spatial 2")

    ## Template-based
    aligned_coords_template = two_d_gpsa(
        n_epochs=N_EPOCHS,
        n_outputs=n_outputs,
        warp_kernel_variance=warp_kernel_variance,
        noise_variance=noise_variance,
        n_latent_gps=n_latent_gps,
        fixed_view_idx=0,
    )
    plt.subplot(133)
    for vv in range(n_views):
        plt.scatter(
            aligned_coords_template[view_idx[vv], 0],
            aligned_coords_template[view_idx[vv], 1],
            c=Y[view_idx[vv], 0],
            marker=markers[vv],
            s=300,
            linewidth=1.8,
            edgecolor="black",
            label="Slice {}".format(vv + 1),
        )
    plt.title(r"$\emph{Template-based}$ alignment")
    plt.xlabel("Spatial 1")
    plt.ylabel("Spatial 2")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("./out/two_d_denovo_vs_templatebased.png")
    plt.show()

    denovo_error = np.mean(
        np.sum(
            (aligned_coords_denovo[view_idx[0]] - aligned_coords_denovo[view_idx[1]])
            ** 2,
            axis=1,
        )
    )
    templatebased_error = np.mean(
        np.sum(
            (
                aligned_coords_template[view_idx[0]]
                - aligned_coords_template[view_idx[1]]
            )
            ** 2,
            axis=1,
        )
    )
    original_error = np.mean(np.sum((X[view_idx[0]] - X[view_idx[1]]) ** 2, axis=1))

    # De novo error: 0.000536963
    # Template error: 0.007253051
    # Observed data error: 0.7329880727046506

    import ipdb

    ipdb.set_trace()
