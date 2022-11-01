import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from os.path import join as pjoin
import scanpy as sc
import squidpy as sq
import anndata
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from gpsa import VariationalGPSA, rbf_kernel
from gpsa.plotting import callback_twod

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern

from scipy.sparse import load_npz

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import r2_score

device = "cuda" if torch.cuda.is_available() else "cpu"


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


DATA_DIR = "../../../data/slideseq/mouse_hippocampus"
N_SAMPLES = 4000

n_spatial_dims = 2
n_views = 2
m_G = 100  # 200
m_X_per_view = 100  # 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 6_000
PRINT_EVERY = 100


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=500)  # 1800
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)

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
data_slice1 = process_data(data_slice1, n_top_genes=3000)


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
data_slice2 = process_data(data_slice2, n_top_genes=3000)


## Remove outlier points outside of puck
MAX_NEIGHBOR_DIST = 700
knn = NearestNeighbors(n_neighbors=10).fit(data_slice1.obsm["spatial"])
neighbor_dists, _ = knn.kneighbors(data_slice1.obsm["spatial"])
inlier_idx = np.where(neighbor_dists[:, -1] < MAX_NEIGHBOR_DIST)[0]
data_slice1 = data_slice1[inlier_idx]

knn = NearestNeighbors(n_neighbors=10).fit(data_slice2.obsm["spatial"])
neighbor_dists, _ = knn.kneighbors(data_slice2.obsm["spatial"])
inlier_idx = np.where(neighbor_dists[:, -1] < MAX_NEIGHBOR_DIST)[0]
data_slice2 = data_slice2[inlier_idx]


## Perform initial coarse adjustment
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


data_slice1 = data_slice1[
    np.random.choice(np.arange(data_slice1.shape[0]), size=N_SAMPLES, replace=False)
]
data_slice2 = data_slice2[
    np.random.choice(np.arange(data_slice2.shape[0]), size=N_SAMPLES, replace=False)
]

sq.gr.spatial_neighbors(data_slice1)
sq.gr.spatial_autocorr(
    data_slice1,
    mode="moran",
)

moran_scores = data_slice1.uns["moranI"]
genes_to_keep = moran_scores.index.values[np.where(moran_scores.I.values > 0.2)[0]]
genes_to_keep = np.intersect1d(genes_to_keep, data_slice2.var.index.values)
N_GENES = len(genes_to_keep)

data_slice1 = data_slice1[:, genes_to_keep]
data_slice2 = data_slice2[:, genes_to_keep]


## Remove genes with low variance
nonzerovar_idx = np.intersect1d(
    np.where(np.array(data_slice1.X.todense()).var(0) > 0.1)[0],
    np.where(np.array(data_slice2.X.todense()).var(0) > 0.1)[0],
)
# import ipdb; ipdb.set_trace()

# data = data[:, nonzerovar_idx]
data_slice1 = data_slice1[:, nonzerovar_idx]
data_slice2 = data_slice2[:, nonzerovar_idx]

assert np.array_equal(data_slice1.var.gene_ids.values, data_slice2.var.gene_ids.values)


all_slices = anndata.concat([data_slice1, data_slice2])
data = data_slice1.concatenate(data_slice2)
# import ipdb; ipdb.set_trace()
n_samples_list = [data[data.obs.batch == str(ii)].shape[0] for ii in range(n_views)]

X1 = np.array(data[data.obs.batch == "0"].obsm["spatial"])
X2 = np.array(data[data.obs.batch == "1"].obsm["spatial"])
Y1 = np.array(data[data.obs.batch == "0"].X.todense())
Y2 = np.array(data[data.obs.batch == "1"].X.todense())

# X1 = np.array(data_slice1.obsm["spatial"])
# X2 = np.array(data_slice2.obsm["spatial"])
# Y1 = np.array(data_slice1.X.todense())
# Y2 = np.array(data_slice2.X.todense())


Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

X = np.concatenate([X1, X2])
Y = np.concatenate([Y1, Y2])

view_idx = [
    np.arange(X1.shape[0]),
    np.arange(X1.shape[0], X1.shape[0] + X2.shape[0]),
]

x = torch.from_numpy(X).float().clone().to(device)
y = torch.from_numpy(Y).float().clone().to(device)


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
    fixed_warp_kernel_variances=np.ones(n_views) * 1e-3,
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

for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0 or t == N_EPOCHS - 1:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))

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

        ax1.invert_yaxis()
        ax2.invert_yaxis()

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
