import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from os.path import join as pjoin
import scanpy as sc
import anndata
import matplotlib.patches as patches

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
landmark_markersize = 200

aligned_coords = pd.read_csv("./out/aligned_coords_slideseq.csv", index_col=0).values
view_idx = pd.read_csv("./out/view_idx_slideseq.csv", index_col=0).values
X = pd.read_csv("./out/X_slideseq.csv", index_col=0).values
Y = pd.read_csv("./out/Y_slideseq.csv", index_col=0).values
data = sc.read_h5ad("./out/data_slideseq.h5")

plt.style.use("dark_background")

# Locations of horn tips
view1_landmark_locs_prealignment = np.array(
    [
        [2.63, -0.28],
        [-0.81, 2.22],
        [4.67, -0.84],
    ]
)
view2_landmark_locs_prealignment = np.array(
    [
        [1.89, 0.24],
        [-0.97, 3.22],
        [5.18, 0.12],
    ]
)

view_idx = []
for vv in range(2):
    view_idx.append(np.where(data.obs.batch.values == str(vv))[0])

close_idx_1 = np.argmin(
    ((X[view_idx[0]] - view1_landmark_locs_prealignment.reshape(3, -1, 2)) ** 2).sum(
        -1
    ),
    axis=1,
)
close_idx_2 = np.argmin(
    ((X[view_idx[1]] - view2_landmark_locs_prealignment.reshape(3, -1, 2)) ** 2).sum(
        -1
    ),
    axis=1,
)

# for vv in range(len(data.obs.batch.unique())):
#     plt.scatter(
#         X[view_idx[vv], 0], X[view_idx[vv], 1], s=1, label="View {}".format(vv + 1)
#     )
# plt.show()
# import ipdb

# ipdb.set_trace()


n_genes = 3
# plt.figure(figsize=(n_genes * 5 + 5, 10), gridspec_kw={'width_ratios': [2., 1, 1, 1]})
fig, ax = plt.subplots(
    2,
    n_genes + 1,
    figsize=(n_genes * 5 + 3, 8),
    gridspec_kw={"width_ratios": [1.1, 1, 1, 1]},
)

# import ipdb

# ipdb.set_trace()

# plt.subplot(2, n_genes + 1, 1)
colors = ["magenta", "cyan"]
plt.sca(ax[0, 0])
for vv in range(len(data.obs.batch.unique())):
    plt.scatter(
        X[view_idx[vv], 0],
        X[view_idx[vv], 1],
        s=1,
        label="Slice {}".format(vv + 1),
        color=colors[vv],
        alpha=0.5,
    )


# for ll in range(len(view1_landmark_locs_prealignment)):
#     plt.scatter(
#         X[view_idx[0]][close_idx_1[ll], 0],
#         X[view_idx[0]][close_idx_1[ll], 1],
#         color="red",
#         marker="*",
#         s=landmark_markersize
#     )

#     plt.scatter(
#         X[view_idx[1]][close_idx_2[ll], 0],
#         X[view_idx[1]][close_idx_2[ll], 1],
#         color="green",
#         marker="*",
#         s=landmark_markersize
#     )


# plt.show()
lgnd = plt.legend(loc="center right", bbox_to_anchor=(-0.05, 0.5))
for handle in lgnd.legendHandles:
    handle.set_sizes([60])
plt.tight_layout()
plt.axis("off")
plt.gca().invert_yaxis()

# plt.subplot(2, n_genes + 1, 5)
plt.sca(ax[1, 0])
for vv in range(len(data.obs.batch.unique())):
    plt.scatter(
        aligned_coords[view_idx[vv], 0],
        aligned_coords[view_idx[vv], 1],
        s=1,
        color=colors[vv],
        alpha=0.5,
    )

# for ll in range(len(view1_landmark_locs_prealignment)):
#     plt.scatter(
#         aligned_coords[view_idx[0]][close_idx_1[ll], 0],
#         aligned_coords[view_idx[0]][close_idx_1[ll], 1],
#         color="red",
#         marker="*",
#         s=landmark_markersize
#     )

#     plt.scatter(
#         aligned_coords[view_idx[1]][close_idx_2[ll], 0],
#         aligned_coords[view_idx[1]][close_idx_2[ll], 1],
#         color="green",
#         marker="*",
#         s=landmark_markersize
#     )

plt.axis("off")
plt.gca().invert_yaxis()


# for gg in range(n_genes):

#     # plt.subplot(2, n_genes + 1, gg + 2)
#     plt.sca(ax[0, gg + 1])
#     for vv in range(len(data.obs.batch.unique())):
#         plt.scatter(
#             X[view_idx[vv], 0],
#             X[view_idx[vv], 1],
#             c=Y[view_idx[vv]][:, gg],
#             s=1,
#         )
#     plt.title(r"$\emph{" + data.var.gene_ids.values[gg].upper() + "}$")
#     plt.axis("off")
#     plt.gca().invert_yaxis()

#     # plt.subplot(2, n_genes + 1, gg + (n_genes + 2) + 1)
#     plt.sca(ax[1, gg + 1])
#     for vv in range(len(data.obs.batch.unique())):
#         plt.scatter(
#             aligned_coords[view_idx[vv], 0],
#             aligned_coords[view_idx[vv], 1],
#             c=Y[view_idx[vv]][:, gg],
#             s=1,
#         )
#     # plt.title(r"$\emph{" + data.var.gene_ids.values[gg] + "}$")
#     plt.axis("off")
#     plt.gca().invert_yaxis()

# # plt.savefig("./out/slideseq_alignment_per_gene.png")
# plt.show()


# import ipdb

# ipdb.set_trace()







gene_names = ["Hpca", "Atp2b1", "Camk2a"]
# gene_idx = np.where(np.isin(data.var.gene_ids.values, gene_names))[0]
gene_idx = [np.where(data.var.gene_ids.values == g)[0] for g in gene_names]
print(gene_idx)

# for gg in range(n_genes):
for ii, gg in enumerate(gene_idx):

    # plt.subplot(2, n_genes + 1, gg + 2)
    plt.sca(ax[0, ii + 1])
    for vv in range(len(data.obs.batch.unique())):
        plt.scatter(
            X[view_idx[vv], 0],
            X[view_idx[vv], 1],
            c=Y[view_idx[vv]][:, gg],
            s=1,
        )
    # plt.title(r"$\emph{" + data.var.gene_ids.values[gg].upper() + "}$")
    plt.title(r"$\emph{" + gene_names[ii].upper() + "}$")
    plt.axis("off")
    plt.gca().invert_yaxis()

    # plt.subplot(2, n_genes + 1, gg + (n_genes + 2) + 1)
    plt.sca(ax[1, ii + 1])
    for vv in range(len(data.obs.batch.unique())):
        plt.scatter(
            aligned_coords[view_idx[vv], 0],
            aligned_coords[view_idx[vv], 1],
            c=Y[view_idx[vv]][:, gg],
            s=1,
        )
    # plt.title(r"$\emph{" + data.var.gene_ids.values[gg] + "}$")
    plt.axis("off")
    plt.gca().invert_yaxis()

plt.savefig("./out/slideseq_alignment_per_gene.png")
plt.show()


import ipdb

ipdb.set_trace()

