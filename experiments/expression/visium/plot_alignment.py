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
import matplotlib.patches as patches

sys.path.append("../../..")
sys.path.append("../../../data")
from plotting.callbacks import callback_oned, callback_twod

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern
from sklearn.model_selection import KFold

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10

aligned_coords = pd.read_csv("./out/aligned_coords_visium.csv", index_col=0).values
view_idx = pd.read_csv("./out/view_idx_visium.csv", index_col=0).values
X = pd.read_csv("./out/X_visium.csv", index_col=0).values
Y = pd.read_csv("./out/Y_visium.csv", index_col=0).values
data = sc.read_h5ad("./out/data_visium.h5")
# import ipdb; ipdb.set_trace()

data_aligned = data.copy()
data_aligned.obsm["spatial"] = aligned_coords


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


GENE_NAME = "mt-Co1"
# ['Pcp2', 'Nrgn', 'Mbp', 'Ddn', 'Camk2a', 'Cbln3', 'Fth1', 'Ttr', 'Gabra6', 'mt-Co1']


# data[data.obs["batch"] == "0"].obsm["spatial"] = scale_spatial_coords(
#     data[data.obs["batch"] == "0"].obsm["spatial"]
# )
# data[data.obs["batch"] == "1"].obsm["spatial"] = scale_spatial_coords(
#     data[data.obs["batch"] == "1"].obsm["spatial"]
# )

data.obsm["spatial"] = np.concatenate(
    [
        scale_spatial_coords(data[data.obs["batch"] == "0"].obsm["spatial"]),
        scale_spatial_coords(data[data.obs["batch"] == "1"].obsm["spatial"]),
    ]
)


xlow = 5.5
xhigh = 8
ylow = 1.1
yhigh = 2.3


fig = plt.figure(figsize=(12, 8), facecolor="white", constrained_layout=True)
ax1 = fig.add_subplot(221, frameon=False)
ax2 = fig.add_subplot(222, frameon=False)
ax3 = fig.add_subplot(223, frameon=False)
ax4 = fig.add_subplot(224, frameon=False)
# ax5 = fig.add_subplot(235, frameon=False)
# ax6 = fig.add_subplot(236, frameon=False)

ax1.scatter(
    data.obsm["spatial"][:, 0],
    data.obsm["spatial"][:, 1],
    # marker="s",
    c=data.obs["total_counts"].values,
    # c=np.array(data[:, GENE_NAME].X.todense()).squeeze(),
    s=1.5,
    alpha=0.5,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
)
ax1.set_title("Unaligned")
ax1.invert_yaxis()


rect = patches.Rectangle(
    (xlow, ylow),
    xhigh - xlow,
    yhigh - ylow,
    linewidth=2,
    edgecolor="r",
    facecolor="none",
)
ax1.add_patch(rect)

ax2.scatter(
    data_aligned.obsm["spatial"][:, 0],
    data_aligned.obsm["spatial"][:, 1],
    # marker="s",
    c=data_aligned.obs["total_counts"].values,
    # c=np.array(data_aligned[:, GENE_NAME].X.todense()).squeeze(),
    s=1.5,
    alpha=0.5,
    # vmin=data_aligned.obs["total_counts"].values.min(),
    # vmax=data_aligned.obs["total_counts"].values.max(),
)
ax2.set_title("Aligned")
ax2.invert_yaxis()

rect = patches.Rectangle(
    (xlow, ylow),
    xhigh - xlow,
    yhigh - ylow,
    linewidth=2,
    edgecolor="r",
    facecolor="none",
)
ax2.add_patch(rect)


data_view1 = data[data.obs["batch"] == "0"]
data_view2 = data[data.obs["batch"] == "1"]


# gpr = GaussianProcessRegressor(kernel=WhiteKernel() + RBF(), normalize_y=True)
# gpr.fit(data_view1.obsm["spatial"], data_view1.obs["total_counts"].values)
# xlimits = [xlow, xhigh]
# ylimits = [ylow, yhigh]
# grid_size = 40
# x1s = np.linspace(*xlimits, num=grid_size)
# x2s = np.linspace(*ylimits, num=grid_size)
# X1, X2 = np.meshgrid(x1s, x2s)
# X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

# preds1 = gpr.predict(X_grid)

# gpr = GaussianProcessRegressor(kernel=WhiteKernel() + RBF(), normalize_y=True)
# gpr.fit(data_view2.obsm["spatial"], data_view2.obs["total_counts"].values)
# xlimits = [xlow, xhigh]
# ylimits = [ylow, yhigh]
# grid_size = 40
# x1s = np.linspace(*xlimits, num=grid_size)
# x2s = np.linspace(*ylimits, num=grid_size)
# X1, X2 = np.meshgrid(x1s, x2s)
# X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

# preds2 = gpr.predict(X_grid)
# preds_diff_unaligned = preds2 - preds1
# ax3.scatter(
#     X_grid[:, 0], X_grid[:, 1], c=preds_diff_unaligned, marker="s", s=20, cmap="bwr"
# )
# ax3.invert_yaxis()

data_view1 = data_view1[
    (data_view1.obsm["spatial"][:, 0] < xhigh)
    & (data_view1.obsm["spatial"][:, 0] > xlow)
    & (data_view1.obsm["spatial"][:, 1] < yhigh)
    & (data_view1.obsm["spatial"][:, 1] > ylow)
]
data_view2 = data_view2[
    (data_view2.obsm["spatial"][:, 0] < xhigh)
    & (data_view2.obsm["spatial"][:, 0] > xlow)
    & (data_view2.obsm["spatial"][:, 1] < yhigh)
    & (data_view2.obsm["spatial"][:, 1] > ylow)
]


ax3.scatter(
    data_view1.obsm["spatial"][:, 0],
    data_view1.obsm["spatial"][:, 1],
    marker="s",
    c=data_view1.obs["total_counts"].values,
    # c=np.array(data_view1[:, GENE_NAME].X.todense()).squeeze(),
    s=100,
    alpha=0.8,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
    label="Slice 1",
)
ax3.scatter(
    data_view2.obsm["spatial"][:, 0],
    data_view2.obsm["spatial"][:, 1],
    marker="o",
    c=data_view2.obs["total_counts"].values,
    # c=np.array(data_view2[:, GENE_NAME].X.todense()).squeeze(),
    s=100,
    alpha=0.8,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
    label="Slice 2",
    edgecolors="black",
    linewidth=2,
)
ax3.invert_yaxis()


data_aligned_view1 = data_aligned[data_aligned.obs["batch"] == "0"]
data_aligned_view2 = data_aligned[data_aligned.obs["batch"] == "1"]


# gpr = GaussianProcessRegressor(kernel=WhiteKernel() + RBF(), normalize_y=True)
# gpr.fit(
#     data_aligned_view1.obsm["spatial"], data_aligned_view1.obs["total_counts"].values
# )
# xlimits = [xlow, xhigh]
# ylimits = [ylow, yhigh]
# grid_size = 40
# x1s = np.linspace(*xlimits, num=grid_size)
# x2s = np.linspace(*ylimits, num=grid_size)
# X1, X2 = np.meshgrid(x1s, x2s)
# X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

# preds1 = gpr.predict(X_grid)

# gpr = GaussianProcessRegressor(kernel=WhiteKernel() + RBF(), normalize_y=True)
# gpr.fit(
#     data_aligned_view2.obsm["spatial"], data_aligned_view2.obs["total_counts"].values
# )
# xlimits = [xlow, xhigh]
# ylimits = [ylow, yhigh]
# grid_size = 40
# x1s = np.linspace(*xlimits, num=grid_size)
# x2s = np.linspace(*ylimits, num=grid_size)
# X1, X2 = np.meshgrid(x1s, x2s)
# X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

# preds2 = gpr.predict(X_grid)
# preds_diff_aligned = preds2 - preds1
# ax6.scatter(
#     X_grid[:, 0],
#     X_grid[:, 1],
#     c=preds_diff_aligned,
#     marker="s",
#     s=20,
#     vmin=preds_diff_unaligned.min(),
#     vmax=preds_diff_unaligned.max(),
#     cmap="bwr",
# )
# ax6.invert_yaxis()

data_aligned_view1 = data_aligned_view1[
    (data_aligned_view1.obsm["spatial"][:, 0] < 8)
    & (data_aligned_view1.obsm["spatial"][:, 0] > 5.5)
    & (data_aligned_view1.obsm["spatial"][:, 1] < 2.3)
    & (data_aligned_view1.obsm["spatial"][:, 1] > 1.1)
]
data_aligned_view2 = data_aligned_view2[
    (data_aligned_view2.obsm["spatial"][:, 0] < 8)
    & (data_aligned_view2.obsm["spatial"][:, 0] > 5.5)
    & (data_aligned_view2.obsm["spatial"][:, 1] < 2.3)
    & (data_aligned_view2.obsm["spatial"][:, 1] > 1.1)
]


ax4.scatter(
    data_aligned_view1.obsm["spatial"][:, 0],
    data_aligned_view1.obsm["spatial"][:, 1],
    marker="s",
    c=data_aligned_view1.obs["total_counts"].values,
    # c=np.array(data_aligned_view1[:, GENE_NAME].X.todense()).squeeze(),
    s=100,
    alpha=0.8,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
    label="Slice 1",
)
ax4.scatter(
    data_aligned_view2.obsm["spatial"][:, 0],
    data_aligned_view2.obsm["spatial"][:, 1],
    marker="o",
    c=data_aligned_view2.obs["total_counts"].values,
    # c=np.array(data_aligned_view2[:, GENE_NAME].X.todense()).squeeze(),
    s=100,
    alpha=0.8,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
    label="Slice 2",
    edgecolors="black",
    linewidth=2,
)
ax4.invert_yaxis()


plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()

plt.savefig("./out/visium_alignment_example.png")
plt.show()

# import ipdb

# ipdb.set_trace()
