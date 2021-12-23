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

aligned_coords_expression = pd.read_csv(
    "./out/multimodal/aligned_coords_expression_visium.csv", index_col=0
).values
aligned_coords_histology = pd.read_csv(
    "./out/multimodal/aligned_coords_histology_visium.csv", index_col=0
).values

view_idx_expression = pd.read_csv(
    "./out/multimodal/view_idx_expression_visium.csv", index_col=0
).values
view_idx_histology = pd.read_csv(
    "./out/multimodal/view_idx_histology_visium.csv", index_col=0
).values

X_expression = pd.read_csv(
    "./out/multimodal/X_expression_visium.csv", index_col=0
).values
X_histology = pd.read_csv("./out/multimodal/X_histology_visium.csv", index_col=0).values

Y_expression = pd.read_csv(
    "./out/multimodal/Y_expression_visium.csv", index_col=0
).values
Y_histology = pd.read_csv(
    "./out/multimodal/Y_histology_rgb_visium.csv", index_col=0
).values

data = sc.read_h5ad("./out/data_visium.h5")


markers = ["o", "X"]

plt.figure(figsize=(12, 10))
for vv in range(2):
    plt.subplot(221)
    plt.title("Unaligned")
    plt.scatter(
        X_histology[view_idx_histology[vv]][:, 0],
        X_histology[view_idx_histology[vv]][:, 1],
        c=Y_histology[view_idx_histology[vv]],
        s=30,
        marker=markers[vv],
    )

    plt.subplot(222)
    plt.title("Aligned")
    plt.scatter(
        aligned_coords_histology[view_idx_histology[vv]][:, 0],
        aligned_coords_histology[view_idx_histology[vv]][:, 1],
        c=Y_histology[view_idx_histology[vv]],
        s=30,
        marker=markers[vv],
    )


box_xlims = [8.0, 10.5]
box_ylims = [4.8, 6.4]


plt.subplot(221)
ax = plt.gca()
rect = patches.Rectangle(
    (box_xlims[0], box_ylims[0]),
    box_xlims[1] - box_xlims[0],
    box_ylims[1] - box_ylims[0],
    linewidth=2,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)

plt.subplot(222)
ax = plt.gca()
rect = patches.Rectangle(
    (box_xlims[0], box_ylims[0]),
    box_xlims[1] - box_xlims[0],
    box_ylims[1] - box_ylims[0],
    linewidth=2,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)


in_idx = np.where(
    (X_histology[:, 0] > box_xlims[0])
    & (X_histology[:, 0] < box_xlims[1])
    & (X_histology[:, 1] > box_ylims[0])
    & (X_histology[:, 1] < box_ylims[1])
)[0]

for vv in range(2):
    plt.subplot(223)
    plt.title("Unaligned")
    curr_idx = np.intersect1d(in_idx, view_idx_histology[vv])
    plt.scatter(
        X_histology[curr_idx][:, 0],
        X_histology[curr_idx][:, 1],
        c=Y_histology[curr_idx],
        s=100,
        marker=markers[vv],
        edgecolors="black" if vv == 1 else None,
        linewidth=0.5,
        alpha=0.8,
    )

    plt.subplot(224)
    plt.title("Aligned")
    plt.scatter(
        aligned_coords_histology[curr_idx][:, 0],
        aligned_coords_histology[curr_idx][:, 1],
        c=Y_histology[curr_idx],
        s=100,
        marker=markers[vv],
        label="Slice {}".format(vv + 1),
        edgecolors="black" if vv == 1 else None,
        linewidth=0.5,
        alpha=0.8,
    )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./out/visium_histology_alignment.png")
plt.close()
# plt.show()


diffs = aligned_coords_histology - X_histology

plt.figure(figsize=(7, 7))
plt.title("Aligned")
for vv in range(2):

    curr_idx = np.intersect1d(in_idx, view_idx_histology[vv])

    if vv == 0:
        plt.scatter(
            aligned_coords_histology[curr_idx][:, 0],
            aligned_coords_histology[curr_idx][:, 1],
            c=Y_histology[curr_idx],
            s=100,
            marker=markers[vv],
            label="Slice {}".format(vv + 1),
            edgecolors="black" if vv == 1 else None,
            linewidth=0.5,
            # alpha=0.8,
        )
    if vv == 1:
        for curr_sub_idx in curr_idx:
            plt.arrow(
                x=X_histology[curr_sub_idx][0],
                y=X_histology[curr_sub_idx][1],
                dx=diffs[curr_sub_idx][0],
                dy=diffs[curr_sub_idx][1],
                width=0.02,
                head_width=0.05,
                color=Y_histology[curr_sub_idx],
                edgecolor="black",
                length_includes_head=True,
                # color="black",
                alpha=0.5,
            )
plt.tight_layout()
plt.savefig("./out/visium_histology_alignment_arrows.png")
plt.show()


import ipdb

ipdb.set_trace()
