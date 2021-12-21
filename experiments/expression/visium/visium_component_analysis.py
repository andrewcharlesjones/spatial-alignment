import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from os.path import join as pjoin
import scanpy as sc
import anndata

sys.path.append("../../..")
sys.path.append("../../../data")
from warps import apply_gp_warp
from util import (
    compute_size_factors,
    poisson_deviance,
    deviance_feature_selection,
    deviance_residuals,
    pearson_residuals,
)
from models.gpsa_vi_lmc import VariationalWarpGP
from plotting.callbacks import callback_oned, callback_twod, callback_twod_aligned_only

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../../paste")
from src.paste import PASTE, visualization

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


DATA_DIR = "../../../data/visium/mouse_brain"
N_GENES = 1000
N_SAMPLES = None

n_spatial_dims = 2
n_views = 2
m_G = 40
m_X_per_view = 40

N_LATENT_GPS = {"expression": 5}


N_EPOCHS = 5000
PRINT_EVERY = 20


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=15000)
    # sc.pp.filter_cells(adata, max_counts=35000)
    adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


data_slice1 = sc.read_visium(pjoin(DATA_DIR, "sample1"))
data_slice1 = process_data(data_slice1)

data_slice2 = sc.read_visium(pjoin(DATA_DIR, "sample2"))
data_slice2 = process_data(data_slice2)

# import ipdb; ipdb.set_trace()

data = data_slice1.concatenate(data_slice2)

errors_union, errors_separate, errors_gpsa = [], [], []

# for repeat_idx in range(N_REPEATS):

if N_SAMPLES is not None:
    rand_idx = np.random.choice(
        np.arange(data_slice1.shape[0]), size=N_SAMPLES, replace=False
    )
    data_slice1 = data_slice1[rand_idx]
    rand_idx = np.random.choice(
        np.arange(data_slice2.shape[0]), size=N_SAMPLES, replace=False
    )
    data_slice2 = data_slice2[rand_idx]

# all_slices = anndata.concat([data_slice1, data_slice2])
n_samples_list = [data_slice1.shape[0], data_slice2.shape[0]]
view_idx = [
    np.arange(data_slice1.shape[0]),
    np.arange(data_slice1.shape[0], data_slice1.shape[0] + data_slice2.shape[0]),
]

# deviances, gene_names = deviance_feature_selection(all_slices.to_df().transpose())
# sorted_idx = np.argsort(-deviances)
# highly_variable_genes = gene_names[sorted_idx][:N_GENES]

# all_slices = all_slices[:, highly_variable_genes]

# import ipdb; ipdb.set_trace()
X1 = data[data.obs.batch == "0"].obsm["spatial"]
X2 = data[data.obs.batch == "1"].obsm["spatial"]
Y1 = np.array(data[data.obs.batch == "0"].X.todense())
Y2 = np.array(data[data.obs.batch == "1"].X.todense())

X1 = scale_spatial_coords(X1)
X2 = scale_spatial_coords(X2)
# import ipdb; ipdb.set_trace()
# Y1 = pearson_residuals(np.array(Y1_unnormalized), theta=100.0)
# Y2 = pearson_residuals(np.array(Y2_unnormalized), theta=100.0)

Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

X = np.concatenate([X1, X2])
Y = np.concatenate([Y1, Y2])
# Y_unnormalized = np.concatenate([Y1_unnormalized, Y2_unnormalized])
# Y_unnormalized_sums = Y_unnormalized.sum(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_outputs = Y.shape[1]

## Drop part of the second view (this is the part we'll try to predict)
x = torch.from_numpy(X).float().clone()
y = torch.from_numpy(Y).float().clone()
# import ipdb; ipdb.set_trace()


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
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    fixed_warp_kernel_variances=np.ones(n_views) * 1.0,
    fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
    # mean_function="identity_initialized",
    # fixed_view_idx=0,
).to(device)

view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train(model, loss_fn, optimizer):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), G_means, F_latent_samples


# Set up figure.
fig = plt.figure(figsize=(15, 6), facecolor="white", constrained_layout=True)
# ax_dict = fig.subplot_mosaic(
#     [
#         ["data", "latent"],
#     ],
# )
ax1 = fig.add_subplot(251, frameon=False)
ax2 = fig.add_subplot(252, frameon=False)
ax3 = fig.add_subplot(253, frameon=False)
ax4 = fig.add_subplot(254, frameon=False)
ax5 = fig.add_subplot(255, frameon=False)

ax6 = fig.add_subplot(256, frameon=False)
ax7 = fig.add_subplot(257, frameon=False)
ax8 = fig.add_subplot(258, frameon=False)
ax9 = fig.add_subplot(259, frameon=False)
ax10 = fig.add_subplot(2, 5, 10, frameon=False)
plt.show(block=False)


for t in range(N_EPOCHS):
    loss, G_means, F_latent_samples = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))

        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ax5.cla()
        ax6.cla()
        ax7.cla()
        ax8.cla()
        ax9.cla()
        ax10.cla()

        ax1.scatter(
            X[view_idx["expression"][0], 0],
            X[view_idx["expression"][0], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][0], 0]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )
        ax2.set_title("Component 1")
        ax2.scatter(
            X[view_idx["expression"][0], 0],
            X[view_idx["expression"][0], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][0], 1]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )
        ax3.set_title("Component 2")
        ax3.scatter(
            X[view_idx["expression"][0], 0],
            X[view_idx["expression"][0], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][0], 2]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )
        ax4.set_title("Component 3")
        ax4.scatter(
            X[view_idx["expression"][0], 0],
            X[view_idx["expression"][0], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][0], 3]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )
        ax5.set_title("Component 4")
        ax5.scatter(
            X[view_idx["expression"][0], 0],
            X[view_idx["expression"][0], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][0], 4]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )
        ax1.set_title("Component 5")

        ## Bottom row

        ax6.scatter(
            X[view_idx["expression"][1], 0],
            X[view_idx["expression"][1], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][1], 0]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )
        ax7.scatter(
            X[view_idx["expression"][1], 0],
            X[view_idx["expression"][1], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][1], 1]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )
        ax8.scatter(
            X[view_idx["expression"][1], 0],
            X[view_idx["expression"][1], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][1], 2]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )
        ax9.scatter(
            X[view_idx["expression"][1], 0],
            X[view_idx["expression"][1], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][1], 3]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )
        ax10.scatter(
            X[view_idx["expression"][1], 0],
            X[view_idx["expression"][1], 1],
            c=F_latent_samples["expression"]
            .mean(0)[view_idx["expression"][1], 4]
            .detach()
            .numpy(),
            marker="H",
            s=10,
        )

        plt.axis("off")
        plt.draw()
        plt.savefig("./out/visium_lowd_components.png")
        plt.pause(1 / 60.0)


plt.close()

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(14, 7))
data_expression_ax = fig.add_subplot(121, frameon=False)
latent_expression_ax = fig.add_subplot(122, frameon=False)
callback_twod(
    model,
    X,
    Y,
    data_expression_ax=data_expression_ax,
    latent_expression_ax=latent_expression_ax,
    X_aligned=G_means,
)
latent_expression_ax.set_title("Aligned data, GPSA")
latent_expression_ax.set_axis_off()
data_expression_ax.set_axis_off()
# plt.axis("off")

plt.tight_layout()
plt.savefig("./out/visium_alignment.png")
# plt.show()
plt.close()
