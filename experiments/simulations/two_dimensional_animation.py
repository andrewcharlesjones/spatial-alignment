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

import matplotlib.animation as animation
import matplotlib.image as mpimg
import os
from os.path import join as pjoin

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 35

n_spatial_dims = 2
n_views = 2
# n_outputs = 10
m_G = 50
m_X_per_view = 50

N_EPOCHS = 2000
PRINT_EVERY = 100
# N_LATENT_GPS = 1


def two_d_gpsa(
    n_outputs,
    n_epochs,
    n_latent_gps,
    warp_kernel_variance=0.1,
    noise_variance=0.0,
    plot_intermediate=True,
    fixed_view_idx=None,
):

    X, Y, n_samples_list, view_idx = generate_twod_data(
        n_views,
        n_outputs,
        grid_size=10,
        n_latent_gps=n_latent_gps["expression"],
        kernel_lengthscale=5.0,
        kernel_variance=warp_kernel_variance,
        noise_variance=noise_variance,
    )
    n_samples_per_view = X.shape[0] // n_views

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
        # fixed_warp_kernel_variances=np.ones(n_views) * 1.0,
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
    data_expression_ax = fig.add_subplot(121, frameon=False)
    latent_expression_ax = fig.add_subplot(122, frameon=False)
    plt.show(block=False)

    loss_trace = []
    error_trace = []
    n_frames = 0
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
            # plt.draw()
            plt.savefig("./tmp/tmp{}".format(n_frames))
            n_frames += 1
            # plt.pause(1 / 60.0)

        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            {"expression": x}, view_idx=view_idx, Ns=Ns
        )

    print("Done!")

    plt.close()

    fig = plt.figure()
    ims = []
    for ii in range(n_frames):
        fname = "./tmp/tmp{}.png".format(ii)
        img = mpimg.imread(fname)
        im = plt.imshow(img)
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_xticks([])
        ims.append([im])
        os.remove(fname)

    writervideo = animation.FFMpegWriter(fps=5)
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=500)
    ani.save(
            pjoin("out", "alignment_animation.mp4"),
            writer=writervideo,
            dpi=1000,
        )
    plt.close()


if __name__ == "__main__":

    n_outputs = 30
    two_d_gpsa(
        n_epochs=N_EPOCHS,
        n_outputs=n_outputs,
        warp_kernel_variance=0.5,
        noise_variance=0.001,
        n_latent_gps={"expression": 5},
        fixed_view_idx=0,
    )


    import ipdb

    ipdb.set_trace()
