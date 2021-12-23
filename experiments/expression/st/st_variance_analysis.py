import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata
import scanpy as sc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

fixed_view_idx = 1
n_views = 4
gene_idx = 0

aligned_coords = pd.read_csv("./out/aligned_coords_st.csv", index_col=0).values
X = pd.read_csv("./out/X_st.csv", index_col=0).values
Y = pd.read_csv("./out/Y_st.csv", index_col=0).values
data = sc.read_h5ad("./out/data_st.h5")

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
    plt.show()

import ipdb

ipdb.set_trace()
