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

# from util import (
#     compute_size_factors,
#     poisson_deviance,
#     deviance_feature_selection,
#     deviance_residuals,
#     pearson_residuals,
#     matern12_kernel,
#     rbf_kernel,
# )
# from models.gpsa_vi_lmc import VariationalWarpGP
# from plotting.callbacks import callback_oned, callback_twod, callback_twod_aligned_only

from scipy.sparse import load_npz


## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches


sys.path.append("../../../../paste")
from src.paste import PASTE, visualization

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import r2_score

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["xtick.labelsize"] = 10
# matplotlib.rcParams["ytick.labelsize"] = 10


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
PRINT_EVERY = 1


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # sc.pp.filter_cells(adata, min_counts=1000)
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


data = data_slice1.concatenate(data_slice2)

shared_gene_names = data.var.gene_ids.index.values
data_knn = data_slice1[:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = np.array(data_knn.X.todense())
Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
preds = knn.predict(X_knn)

# preds = Y_knn[indices[:, 1]]
r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

sorted_idx = np.argsort(r2_vals)

n_genes_to_plot = 3
best_genes = data_knn.var.gene_ids.index.values[sorted_idx[-n_genes_to_plot:]]
worst_genes = data_knn.var.gene_ids.index.values[sorted_idx[:n_genes_to_plot]]


plt.figure(figsize=(15, 10))

for ii in range(n_genes_to_plot):

    plt.subplot(2, n_genes_to_plot, ii + 1)
    plt.scatter(
        data_slice1.obsm["spatial"][:, 0],
        data_slice1.obsm["spatial"][:, 1],
        c=np.array(data_knn[:, best_genes[ii]].X.todense()).squeeze(),
        s=1,
        marker="H",
    )
    plt.title(r"$\emph{" + best_genes[ii] + "}$")
    plt.axis("off")

    plt.subplot(2, n_genes_to_plot, n_genes_to_plot + (ii + 1))
    plt.scatter(
        data_slice1.obsm["spatial"][:, 0],
        data_slice1.obsm["spatial"][:, 1],
        c=np.array(data_knn[:, worst_genes[ii]].X.todense()).squeeze(),
        s=1,
        marker="H",
    )
    plt.title(r"$\emph{" + worst_genes[ii] + "}$")
    plt.axis("off")

plt.savefig("./out/slideseq_spatially_variable_genes.png")
plt.show()


import ipdb

ipdb.set_trace()
