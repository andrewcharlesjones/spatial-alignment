import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata
import scanpy as sc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import r2_score


import sys

sys.path.append("../../../data")
from st.load_st_data import load_st_data

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


DATA_DIR = "../../../data/st/"
N_GENES = 20
N_SAMPLES = None
N_LAYERS = 4
fixed_view_idx = 1

fixed_view_idx = 1
n_views = 4


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

gene_idx_to_keep = np.where(r2_vals > 0.0)[0]
N_GENES = min(N_GENES, len(gene_idx_to_keep))
gene_names_to_keep = data_knn.var.gene_ids.index.values[gene_idx_to_keep]
gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
r2_vals_sorted = -1 * np.sort(-r2_vals[gene_idx_to_keep])
# if N_GENES < len(gene_names_to_keep):
#     gene_names_to_keep = gene_names_to_keep[:N_GENES]
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


X_full = np.concatenate(X_list)
Y_full = np.concatenate(Y_list)


aligned_coords = pd.read_csv("./out/aligned_coords_st.csv", index_col=0).values
X = pd.read_csv("./out/X_st.csv", index_col=0).values
Y = pd.read_csv("./out/Y_st.csv", index_col=0).values
# data = sc.read_h5ad("./out/data_st.h5")

# import ipdb; ipdb.set_trace()
assert np.allclose(Y, Y_full[:, : Y.shape[1]], atol=1e-6)
Y = Y_full.copy()

view_idx = []
for vv in data.obs.batch.unique():
    view_idx.append(np.where(data.obs.batch.values == vv)[0])

## Smoothed over template frame
min_spatial_x, min_spatial_y = X[view_idx[fixed_view_idx]].min(0)
max_spatial_x, max_spatial_y = X[view_idx[fixed_view_idx]].max(0)

# Make grid
X_grid = X[view_idx[fixed_view_idx]].copy()

fig, ax = plt.subplots(
    1, n_views, gridspec_kw={"width_ratios": [1, 1, 1, 1.25]}, figsize=(22, 5)
)
smoothed_outputs = []
for vv in range(n_views):
    if vv == fixed_view_idx:
        X_input = X[view_idx[vv]]
    else:
        X_input = aligned_coords[view_idx[vv]]
    # model = GaussianProcessRegressor(WhiteKernel() + RBF())
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_input, Y[view_idx[vv]])
    preds = model.predict(X_grid)

    smoothed_outputs.append(np.expand_dims(preds, 0))


smoothed_outputs = np.concatenate(smoothed_outputs, axis=0)
# variances = np.var(smoothed_outputs, axis=0)
# vmin_exp, vmax_exp = smoothed_outputs[:, :, :4].min(), smoothed_outputs[:, :, :4].max()
# vmin_var, vmax_var = variances[:, :4].min(), variances[:, :4].max()


n_neighbors = 5
nn = NearestNeighbors(n_neighbors=n_neighbors)
nn.fit(X_grid)
_, neighbor_idx = nn.kneighbors(X_grid)


neighbor_data = []
for vv in range(n_views):
    curr_view_neighbors = smoothed_outputs[vv][neighbor_idx].transpose(1, 0, 2)
    neighbor_data.append(curr_view_neighbors)

neighbor_data = np.concatenate(neighbor_data)
variances = np.var(neighbor_data, axis=0)

vmin_exp, vmax_exp = smoothed_outputs[:, :, :4].min(), smoothed_outputs[:, :, :4].max()
vmin_var, vmax_var = variances[:, :4].min(), variances[:, :4].max()

sorted_idx = np.argsort(-np.mean(variances, axis=0))
# import ipdb; ipdb.set_trace()

for vv in range(n_views):

    ax[vv].set_facecolor("lightgray")
    ax[vv].get_xaxis().set_ticks([])
    ax[vv].get_yaxis().set_ticks([])

    curr_gene_idx = sorted_idx[vv]
    plt.sca(ax[vv])

    plt.scatter(
        X_grid[:, 0],
        X_grid[:, 1],
        c=variances[:, curr_gene_idx],
        marker="o",
        s=60,
        vmin=vmin_var,
        vmax=vmax_var,
    )
    plt.title(r"$\emph{" + data.var.gene_ids[curr_gene_idx] + "}$ variance")

plt.colorbar()
plt.savefig("./out/st_aligned_smoothed_and_variance.png")
# plt.show()
plt.close()

GENE_IDX = 1

vmin, vmax = (
    smoothed_outputs[:, :, GENE_IDX].min(),
    smoothed_outputs[:, :, GENE_IDX].max(),
)
for vv in range(n_views):
    plt.figure(figsize=(5, 5))
    plt.scatter(
        X_grid[:, 0],
        X_grid[:, 1],
        c=smoothed_outputs[vv][:, GENE_IDX],
        marker="o",
        s=60,
        vmin=vmin,
        vmax=vmax,
    )
    plt.axis("off")
    plt.savefig(
        "./out/st_slice_smoothed_{}_slice{}.png".format(
            data.var.gene_ids[GENE_IDX], vv + 1
        )
    )
    plt.close()


GENE_IDX_TO_PLOT = sorted_idx[0]
SPATIAL_IDX_TO_PLOT = [20, 50, 70]
neighbor_output_mat = smoothed_outputs[:, neighbor_idx, :]

for curr_spatial_idx in SPATIAL_IDX_TO_PLOT:
    print(X_grid[curr_spatial_idx])
    plt.figure(figsize=(5, 5))
    for vv in range(n_views):
        # import ipdb; ipdb.set_trace()
        xs = [vv + 1] * n_neighbors
        # xs += np.random.normal(scale=0.05, size=len(xs))
        ys = neighbor_output_mat[vv, curr_spatial_idx, :, GENE_IDX_TO_PLOT]
        plt.scatter(xs, ys, color="black")
    plt.xticks(np.arange(1, 5))

    plt.xlabel("Slice index")
    plt.ylabel(r"$\emph{" + data.var.gene_ids[GENE_IDX_TO_PLOT] + "}$ expression")
    plt.tight_layout()
    plt.savefig("./out/st_variance_scatterplot_loc{}.png".format(curr_spatial_idx))
    # plt.show()
    plt.close()


pd.DataFrame(variances.mean(0).reshape(-1, 1), index=data.var.gene_ids).to_csv(
    "./out/st_avg_gene_variances.csv"
)


plt.figure(figsize=(7, 7))
variance_df = pd.melt(pd.DataFrame(variances[:, np.argsort(variances.mean(0))]))
variance_df.variable = variance_df.variable + 1
sns.lineplot(data=variance_df, x="variable", y="value")
plt.xlabel("Gene index (sorted)")
plt.ylabel("Variance")
plt.title(r"$z$-axis variance")
plt.tight_layout()
plt.savefig("./out/st_genewise_avg_spatial_variance.png")
plt.show()

import ipdb

ipdb.set_trace()
