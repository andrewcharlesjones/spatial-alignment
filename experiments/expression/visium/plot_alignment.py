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
# from plotting.callbacks import callback_oned, callback_twod
from gpsa.plotting import callback_oned, callback_twod

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


plt.style.use("dark_background")
# plt.rcParams.update({
#     "lines.color": "white",
#     "patch.edgecolor": "white",
#     "text.color": "white",
#     "axes.facecolor": "lightgray",
#     "axes.edgecolor": "lightgray",
#     "axes.labelcolor": "white",
#     "xtick.color": "white",
#     "ytick.color": "white",
#     "grid.color": "lightgray",
#     "figure.facecolor": "lightgray",
#     "figure.edgecolor": "lightgray",
#     "savefig.facecolor": "lightgray",
#     "savefig.edgecolor": "lightgray"})

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


# fig = plt.figure(figsize=(24, 5), facecolor="white", constrained_layout=True)
# ax1 = fig.add_subplot(141, frameon=False)
# ax2 = fig.add_subplot(142, frameon=False)
# ax3 = fig.add_subplot(143, frameon=False)
# ax4 = fig.add_subplot(144, frameon=False)
# ax5 = fig.add_subplot(235, frameon=False)
# ax6 = fig.add_subplot(236, frameon=False)

fig, ax = plt.subplots(
    1,
    4,
    figsize=(23, 5),
    # gridspec_kw={"width_ratios": [1, 1, 1, 1]},
)

# plt.sca(ax1)
plt.sca(ax[0])
plt.xticks([])
plt.yticks([])

plt.scatter(
    data.obsm["spatial"][:, 0],
    data.obsm["spatial"][:, 1],
    # marker="s",
    c=data.obs["total_counts"].values,
    # c=np.array(data[:, GENE_NAME].X.todense()).squeeze(),
    s=5,
    alpha=0.5,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
    cmap="coolwarm",
)
plt.title("Unaligned")
plt.gca().invert_yaxis()


rect = patches.Rectangle(
    (xlow, ylow),
    xhigh - xlow,
    yhigh - ylow,
    linewidth=2,
    edgecolor="lime",
    facecolor="none",
)
plt.gca().add_patch(rect)

plt.sca(ax[1])
plt.xticks([])
plt.yticks([])

plt.scatter(
    data_aligned.obsm["spatial"][:, 0],
    data_aligned.obsm["spatial"][:, 1],
    # marker="s",
    c=data_aligned.obs["total_counts"].values,
    # c=np.array(data_aligned[:, GENE_NAME].X.todense()).squeeze(),
    s=5,
    alpha=0.5,
    # vmin=data_aligned.obs["total_counts"].values.min(),
    # vmax=data_aligned.obs["total_counts"].values.max(),
    cmap="coolwarm",
)
plt.gca().set_title("Aligned")
plt.gca().invert_yaxis()

rect = patches.Rectangle(
    (xlow, ylow),
    xhigh - xlow,
    yhigh - ylow,
    linewidth=2,
    edgecolor="lime",
    facecolor="none",
)
plt.gca().add_patch(rect)


data_view1 = data[data.obs["batch"] == "0"]
data_view2 = data[data.obs["batch"] == "1"]


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


plt.sca(ax[2])
plt.xticks([])
plt.yticks([])

plt.scatter(
    data_view1.obsm["spatial"][:, 0],
    data_view1.obsm["spatial"][:, 1],
    marker="s",
    c=data_view1.obs["total_counts"].values,
    # c=np.array(data_view1[:, GENE_NAME].X.todense()).squeeze(),
    s=200,
    alpha=0.8,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
    label="Slice 1",
    cmap="coolwarm",
)
plt.scatter(
    data_view2.obsm["spatial"][:, 0],
    data_view2.obsm["spatial"][:, 1],
    marker="o",
    c=data_view2.obs["total_counts"].values,
    # c=np.array(data_view2[:, GENE_NAME].X.todense()).squeeze(),
    s=200,
    alpha=0.8,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
    label="Slice 2",
    # edgecolors="black",
    linewidth=2,
    cmap="coolwarm",
)
plt.gca().invert_yaxis()
plt.gca().set_title("Unaligned")


data_aligned_view1 = data_aligned[data_aligned.obs["batch"] == "0"]
data_aligned_view2 = data_aligned[data_aligned.obs["batch"] == "1"]


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


plt.sca(ax[3])
plt.xticks([])
plt.yticks([])

plt.scatter(
    data_aligned_view1.obsm["spatial"][:, 0],
    data_aligned_view1.obsm["spatial"][:, 1],
    marker="s",
    c=data_aligned_view1.obs["total_counts"].values,
    # c=np.array(data_aligned_view1[:, GENE_NAME].X.todense()).squeeze(),
    s=200,
    alpha=0.8,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
    label="Slice 1",
    cmap="coolwarm",
)
plt.scatter(
    data_aligned_view2.obsm["spatial"][:, 0],
    data_aligned_view2.obsm["spatial"][:, 1],
    marker="o",
    c=data_aligned_view2.obs["total_counts"].values,
    # c=np.array(data_aligned_view2[:, GENE_NAME].X.todense()).squeeze(),
    s=200,
    alpha=0.8,
    # vmin=data.obs["total_counts"].values.min(),
    # vmax=data.obs["total_counts"].values.max(),
    label="Slice 2",
    # edgecolors="black",
    linewidth=2,
    cmap="coolwarm",
)
plt.gca().invert_yaxis()
plt.gca().set_title("Aligned")


plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()

plt.savefig("./out/visium_alignment_example.png")
# ax1.set_facecolor("lightgray")
# ax2.set_facecolor("lightgray")
# ax3.set_facecolor("lightgray")
# ax4.set_facecolor("lightgray")
plt.show()

# import ipdb

# ipdb.set_trace()
