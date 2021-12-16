import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from os.path import join as pjoin
import scanpy as sc
import anndata
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append("../../..")
sys.path.append("../../../data")
from util import (
    compute_size_factors,
    poisson_deviance,
    deviance_feature_selection,
    deviance_residuals,
    pearson_residuals,
)
from util import matern12_kernel, matern32_kernel, rbf_kernel
from models.gpsa_vi_lmc import VariationalWarpGP
from plotting.callbacks import callback_oned, callback_twod

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../../paste")
from src.paste import PASTE, visualization

from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor


def scale_spatial_coords(X, max_val=10):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


DATA_DIR = "../../../data/visium/mouse_brain"
N_GENES = 500
N_SAMPLES = None
N_REPEATS = 10
predict_idx = np.arange(10)
frac_test = 0.2
frac_train = 1 - frac_test

n_spatial_dims = 2
n_views = 2
m_G = 200
m_X_per_view = 200

N_EPOCHS = 2000
PRINT_EVERY = 50
N_LATENT_GPS = {"expression": 20}

spatial_locs_sample1_path = pjoin(
    DATA_DIR, "sample1", "filtered_feature_bc_matrix/spatial_locs_small.csv"
)
data_sample1_path = pjoin(
    DATA_DIR, "sample1", "filtered_feature_bc_matrix/gene_expression_small.csv"
)
X_df_sample1 = pd.read_csv(spatial_locs_sample1_path, index_col=0)
X_orig1 = np.vstack([X_df_sample1.array_col.values, X_df_sample1.array_row.values]).T
X_orig1 = scale_spatial_coords(X_orig1)
slice1 = sc.read_csv(data_sample1_path)
slice1.obsm["spatial"] = X_orig1
sc.pp.filter_genes(slice1, min_counts=15)
sc.pp.filter_cells(slice1, min_counts=100)

spatial_locs_sample2_path = pjoin(
    DATA_DIR, "sample2", "filtered_feature_bc_matrix/spatial_locs_small.csv"
)
data_sample2_path = pjoin(
    DATA_DIR, "sample2", "filtered_feature_bc_matrix/gene_expression_small.csv"
)
X_df_sample2 = pd.read_csv(spatial_locs_sample2_path, index_col=0)
X_orig2 = np.vstack([X_df_sample2.array_col.values, X_df_sample2.array_row.values]).T
X_orig2 = scale_spatial_coords(X_orig2)
slice2 = sc.read_csv(data_sample2_path)
slice2.obsm["spatial"] = X_orig2
sc.pp.filter_genes(slice2, min_counts=15)
sc.pp.filter_cells(slice2, min_counts=100)

errors_union, errors_separate, errors_gpsa = [], [], []

# for repeat_idx in range(N_REPEATS):


if N_SAMPLES is not None:
    rand_idx = np.random.choice(
        np.arange(slice1.shape[0]), size=N_SAMPLES, replace=False
    )
    slice1 = slice1[rand_idx]
    rand_idx = np.random.choice(
        np.arange(slice2.shape[0]), size=N_SAMPLES, replace=False
    )
    slice2 = slice2[rand_idx]


all_slices = anndata.concat([slice1, slice2])
n_samples_list = [slice1.shape[0], slice2.shape[0]]
view_idx = [
    np.arange(slice1.shape[0]),
    np.arange(slice1.shape[0], slice1.shape[0] + slice2.shape[0]),
]


deviances, gene_names = deviance_feature_selection(all_slices.to_df().transpose())
sorted_idx = np.argsort(-deviances)
highly_variable_genes = gene_names[sorted_idx][:N_GENES]

# highly_variable_genes = gene_names[sorted_idx][1:4]
all_slices = all_slices[:, highly_variable_genes]


X1 = all_slices.obsm["spatial"][: slice1.shape[0]]
X2 = all_slices.obsm["spatial"][slice1.shape[0] :]
Y1_unnormalized = all_slices.X[: slice1.shape[0]]
Y2_unnormalized = all_slices.X[slice1.shape[0] :]
Y1 = pearson_residuals(np.array(Y1_unnormalized), theta=100.0)
Y2 = pearson_residuals(np.array(Y2_unnormalized), theta=100.0)

Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

X = np.concatenate([X1, X2])
Y = np.concatenate([Y1, Y2])

device = "cuda" if torch.cuda.is_available() else "cpu"


n_outputs = all_slices.shape[1]

nearestneighbors = KNeighborsRegressor(n_neighbors=1, weights="distance")

nearestneighbors.fit(X1, Y1)
Y2_smoothed = nearestneighbors.predict(X2)

Y_diffs = Y2 - Y2_smoothed
gene_idx = 3


plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.title("Sample 1")
# plt.scatter(grid[:, 0], grid[:, 1], c=Y1_gridded[:, :, gene_idx].ravel())
plt.scatter(X1[:, 0], X1[:, 1], c=Y1[:, gene_idx], s=7, marker="h")
plt.colorbar()
plt.subplot(132)
plt.title("Sample 2")
# plt.scatter(grid[:, 0], grid[:, 1], c=Y1_gridded[:, :, gene_idx].ravel())
plt.scatter(X2[:, 0], X2[:, 1], c=Y2[:, gene_idx], s=7, marker="h")
plt.colorbar()
plt.subplot(133)
plt.title("Difference")
plt.scatter(
    X2[:, 0],
    X2[:, 1],
    c=Y_diffs[:, gene_idx],
    cmap="bwr",
    s=7,
)
plt.colorbar()
plt.savefig("./out/visium_difference_one_gene.png")
plt.show()


plt.show()
import ipdb

ipdb.set_trace()
