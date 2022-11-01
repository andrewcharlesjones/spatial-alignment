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

# from models.gpsa_vi_lmc import VariationalWarpGP
# from plotting.callbacks import callback_oned, callback_twod, callback_twod_aligned_only
from gpsa import VariationalGPSA, matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod

# sys.path.append("../../../data/st/")
from st.load_st_data import load_st_data

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


DATA_DIR = "../../../data/st/"
N_GENES = 20
N_SAMPLES = None
N_LAYERS = 4
fixed_view_idx = 1

n_spatial_dims = 2
n_views = 4
m_G = 200
m_X_per_view = 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 5000
PRINT_EVERY = 25


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=100)
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    # sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


data_slice1, data_slice2, data_slice3, data_slice4 = load_st_data(
    layers=np.arange(N_LAYERS) + 1
)
process_data(data_slice1, n_top_genes=3000)
process_data(data_slice2, n_top_genes=3000)
process_data(data_slice3, n_top_genes=3000)
process_data(data_slice4, n_top_genes=3000)


## Save original data
plt.figure(figsize=(20, 5))

for ii, curr_slice in enumerate([data_slice1, data_slice2, data_slice3, data_slice4]):
    plt.subplot(1, 4, ii + 1)
    plt.scatter(
        curr_slice.obsm["spatial"][:, 0], curr_slice.obsm["spatial"][:, 1], s=30
    )
    plt.title("Slice {}".format(ii + 1), fontsize=30)
    plt.axis("off")

plt.savefig("./out/st_original_slices.png")
# plt.show()
plt.close()


data = anndata.AnnData.concatenate(data_slice1, data_slice2, data_slice3, data_slice4)

# plt.figure(figsize=(5, 5))
# plt.scatter(data[data.obs["batch"] == "0"].obsm["spatial"][:, 0], data[data.obs["batch"] == "0"].obsm["spatial"][:, 1])
# plt.scatter(data[data.obs["batch"] == "1"].obsm["spatial"][:, 0], data[data.obs["batch"] == "1"].obsm["spatial"][:, 1])
# plt.show()
# import ipdb; ipdb.set_trace()


shared_gene_names = data.var.gene_ids.index.values
data_knn = data_slice1[:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = data_knn.X
Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
# nbrs = NearestNeighbors(n_neighbors=2).fit(X_knn)
# distances, indices = nbrs.kneighbors(X_knn)
knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
preds = knn.predict(X_knn)

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
#     sc.pl.spatial(data_knn, img_key=None, color=[gene_name], spot_size=1)


n_samples_list = [
    data_slice1.shape[0],
    data_slice2.shape[0],
    data_slice3.shape[0],
    data_slice4.shape[0],
]
cumulative_sum = np.cumsum(n_samples_list)
cumulative_sum = np.insert(cumulative_sum, 0, 0)
view_idx = [
    np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
]

X_list = []
Y_list = []
for vv in range(n_views):
    curr_X = np.array(data[data.obs.batch == str(vv)].obsm["spatial"])
    curr_Y = data[data.obs.batch == str(vv)].X

    curr_X = scale_spatial_coords(curr_X)
    curr_Y = (curr_Y - curr_Y.mean(0)) / curr_Y.std(0)

    X_list.append(curr_X)
    Y_list.append(curr_Y)


X = np.concatenate(X_list)
Y = np.concatenate(Y_list)

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

model = VariationalGPSA(
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
    fixed_view_idx=fixed_view_idx,
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
plt.show(block=False)


gene_idx = 0

for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        ax1.cla()
        ax2.cla()

        curr_aligned_coords = G_means["expression"].detach().numpy()
        curr_aligned_coords_slice1 = curr_aligned_coords[view_idx["expression"][0]]
        curr_aligned_coords_slice2 = curr_aligned_coords[view_idx["expression"][1]]
        curr_aligned_coords_slice3 = curr_aligned_coords[view_idx["expression"][2]]

        for vv, curr_X in enumerate(X_list):
            ax1.scatter(curr_X[:, 0], curr_X[:, 1], alpha=0.5)

            if vv == fixed_view_idx:
                ax1.scatter(curr_X[:, 0], curr_X[:, 1], alpha=0.5, color="black")
                ax2.scatter(
                    X_list[vv][:, 0], X_list[vv][:, 1], alpha=0.5, color="black"
                )

            ax2.scatter(
                curr_aligned_coords[view_idx["expression"][vv]][:, 0],
                curr_aligned_coords[view_idx["expression"][vv]][:, 1],
                alpha=0.5,
            )

        plt.draw()
        plt.savefig("./out/st_alignment.png")
        plt.pause(1 / 60.0)

        pd.DataFrame(curr_aligned_coords).to_csv("./out/aligned_coords_st.csv")
        pd.DataFrame(view_idx["expression"]).to_csv("./out/view_idx_st.csv")
        pd.DataFrame(X).to_csv("./out/X_st.csv")
        pd.DataFrame(Y).to_csv("./out/Y_st.csv")
        data.write("./out/data_st.h5")

        if model.n_latent_gps["expression"] is not None:
            curr_W = model.W_dict["expression"].detach().numpy()
            pd.DataFrame(curr_W).to_csv("./out/W_st.csv")


plt.close()
