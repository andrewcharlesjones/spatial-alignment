import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from os.path import join as pjoin
import scanpy as sc
import anndata
import time

sys.path.append("../../..")
sys.path.append("../../../data")
from warps import apply_gp_warp
from util import (
    compute_size_factors,
    poisson_deviance,
    deviance_feature_selection,
    deviance_residuals,
    pearson_residuals,
    matern12_kernel,
    rbf_kernel,
)
from models.gpsa_vi_lmc import VariationalWarpGP
from plotting.callbacks import callback_oned, callback_twod, callback_twod_aligned_only

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from scipy.sparse import load_npz

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches


sys.path.append("../../../../paste")
from src.paste import PASTE, visualization

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import r2_score


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


DATA_DIR = "../../../data/slideseq/mouse_hippocampus"
N_GENES = 10
N_SAMPLES = None

n_spatial_dims = 2
n_views = 2
m_G = 200
m_X_per_view = 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 5000
PRINT_EVERY = 50


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=1000)  # 1800
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    # sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


spatial_locs_slice1 = pd.read_csv(
    pjoin(DATA_DIR, "Puck_200115_08_spatial_locs.csv"), index_col=0
)
expression_slice1 = load_npz(pjoin(DATA_DIR, "Puck_200115_08_expression.npz"))
gene_names_slice1 = pd.read_csv(
    pjoin(DATA_DIR, "Puck_200115_08_gene_names.csv"), index_col=0
)
barcode_names_slice1 = pd.read_csv(
    pjoin(DATA_DIR, "Puck_200115_08_barcode_names.csv"), index_col=0
)

data_slice1 = anndata.AnnData(
    X=expression_slice1, obs=barcode_names_slice1, var=gene_names_slice1
)
data_slice1.obsm["spatial"] = spatial_locs_slice1.values
data_slice1 = process_data(data_slice1, n_top_genes=6000)


spatial_locs_slice2 = pd.read_csv(
    pjoin(DATA_DIR, "Puck_191204_01_spatial_locs.csv"), index_col=0
)
expression_slice2 = load_npz(pjoin(DATA_DIR, "Puck_191204_01_expression.npz"))
gene_names_slice2 = pd.read_csv(
    pjoin(DATA_DIR, "Puck_191204_01_gene_names.csv"), index_col=0
)
barcode_names_slice2 = pd.read_csv(
    pjoin(DATA_DIR, "Puck_191204_01_barcode_names.csv"), index_col=0
)

data_slice2 = anndata.AnnData(
    X=expression_slice2, obs=barcode_names_slice2, var=gene_names_slice2
)
data_slice2.obsm["spatial"] = spatial_locs_slice2.values
data_slice2 = process_data(data_slice2, n_top_genes=6000)


## Remove outlier points outside of puck
knn = NearestNeighbors(n_neighbors=10).fit(data_slice1.obsm["spatial"])
neighbor_dists, _ = knn.kneighbors(data_slice1.obsm["spatial"])
inlier_idx = np.where(neighbor_dists[:, -1] < 700)[0]
data_slice1 = data_slice1[inlier_idx]

knn = NearestNeighbors(n_neighbors=10).fit(data_slice2.obsm["spatial"])
neighbor_dists, _ = knn.kneighbors(data_slice2.obsm["spatial"])
inlier_idx = np.where(neighbor_dists[:, -1] < 700)[0]
data_slice2 = data_slice2[inlier_idx]
# import ipdb; ipdb.set_trace()


## Save original data
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(
    data_slice1.obsm["spatial"][:, 0],
    data_slice1.obsm["spatial"][:, 1],
    # c=np.log(np.array(data_slice1.X[:, 0].todense()) + 1)
    s=3,
)
plt.title("Slice 1", fontsize=30)
plt.gca().invert_yaxis()
plt.axis("off")

plt.subplot(122)
plt.scatter(
    data_slice2.obsm["spatial"][:, 0],
    data_slice2.obsm["spatial"][:, 1],
    # c=np.log(np.array(data_slice2.X[:, 0].todense()) + 1)
    s=3,
)
plt.title("Slice 2", fontsize=30)
plt.gca().invert_yaxis()
plt.axis("off")
plt.savefig("./out/slideseq_original_slices.png")
# plt.show()
plt.close()
# import ipdb

# ipdb.set_trace()


angle = 1.45
slice1_coords = data_slice1.obsm["spatial"].copy()
slice2_coords = data_slice2.obsm["spatial"].copy()
slice1_coords = scale_spatial_coords(slice1_coords, max_val=10) - 5
slice2_coords = scale_spatial_coords(slice2_coords, max_val=10) - 5

R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
slice2_coords = slice2_coords @ R

slice2_coords += np.array([1.0, 1.0])

data_slice1.obsm["spatial"] = slice1_coords
data_slice2.obsm["spatial"] = slice2_coords

print(data_slice1.shape, data_slice2.shape)

data = data_slice1.concatenate(data_slice2)

shared_gene_names = data.var.gene_ids.index.values
data_knn = data_slice1[:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = np.array(data_knn.X.todense())
Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
# nbrs = NearestNeighbors(n_neighbors=2).fit(X_knn)
# distances, indices = nbrs.kneighbors(X_knn)
knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
preds = knn.predict(X_knn)

# preds = Y_knn[indices[:, 1]]
r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

gene_idx_to_keep = np.where(r2_vals > 0.3)[0]
N_GENES = min(N_GENES, len(gene_idx_to_keep))
gene_names_to_keep = data_knn.var.gene_ids.index.values[gene_idx_to_keep]
gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
r2_vals_sorted = -1 * np.sort(-r2_vals[gene_idx_to_keep])
if N_GENES < len(gene_names_to_keep):
    gene_names_to_keep = gene_names_to_keep[:N_GENES]
data = data[:, gene_names_to_keep]


# for ii, gene_name in enumerate(gene_names_to_keep):
#     print(r2_vals_sorted[ii], flush=True)
#     plt.figure(figsize=(10, 5))
#     plt.subplot(121)
#     ax = plt.gca()
#     sc.pl.spatial(data_slice1, img_key=None, color=[gene_name], spot_size=0.1, ax=ax, show=False)

#     plt.subplot(122)
#     ax = plt.gca()
#     sc.pl.spatial(data_slice2, img_key=None, color=[gene_name], spot_size=0.1, ax=ax, show=False)

#     plt.show()


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

X1 = np.array(data[data.obs.batch == "0"].obsm["spatial"])
X2 = np.array(data[data.obs.batch == "1"].obsm["spatial"])
Y1 = np.array(data[data.obs.batch == "0"].X.todense())
Y2 = np.array(data[data.obs.batch == "1"].X.todense())

# X1 = scale_spatial_coords(X1)
# X2 = scale_spatial_coords(X2)

Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

X = np.concatenate([X1, X2])
Y = np.concatenate([Y1, Y2])

device = "cuda" if torch.cuda.is_available() else "cpu"

n_outputs = Y.shape[1]

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
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    kernel_func_warp=rbf_kernel,
    kernel_func_data=rbf_kernel,
    # fixed_warp_kernel_variances=np.ones(n_views) * 1.,
    # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
    fixed_view_idx=0,
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

    return loss.item(), G_means


# Set up figure.
fig = plt.figure(figsize=(10, 5), facecolor="white", constrained_layout=True)
ax1 = fig.add_subplot(121, frameon=False)
ax2 = fig.add_subplot(122, frameon=False)
ax1.invert_yaxis()
ax2.invert_yaxis()
plt.show(block=False)


gene_idx = 0

for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        # diff_expression_ax.cla()
        ax1.cla()
        ax2.cla()

        curr_aligned_coords = G_means["expression"].detach().numpy()
        curr_aligned_coords_slice1 = curr_aligned_coords[view_idx["expression"][0]]
        curr_aligned_coords_slice2 = curr_aligned_coords[view_idx["expression"][1]]

        ax1.scatter(X1[:, 0], X1[:, 1], alpha=0.3)
        ax1.scatter(X2[:, 0], X2[:, 1], alpha=0.3)

        ax2.scatter(X1[:, 0], X1[:, 1], alpha=0.3)
        ax2.scatter(
            curr_aligned_coords_slice2[:, 0],
            curr_aligned_coords_slice2[:, 1],
            alpha=0.3,
        )

        plt.draw()
        plt.savefig("./out/slideseq_alignment.png")
        plt.pause(1 / 60.0)

        pd.DataFrame(curr_aligned_coords).to_csv("./out/aligned_coords_slideseq.csv")
        pd.DataFrame(view_idx["expression"]).to_csv("./out/view_idx_slideseq.csv")
        pd.DataFrame(X).to_csv("./out/X_slideseq.csv")
        pd.DataFrame(Y).to_csv("./out/Y_slideseq.csv")
        data.write("./out/data_slideseq.h5")

        if model.n_latent_gps["expression"] is not None:
            curr_W = model.W_dict["expression"].detach().numpy()
            pd.DataFrame(curr_W).to_csv("./out/W_slideseq.csv")


plt.close()
