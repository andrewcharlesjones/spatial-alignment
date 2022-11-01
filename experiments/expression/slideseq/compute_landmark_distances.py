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
from sklearn.metrics import pairwise_distances

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["xtick.labelsize"] = 10
# matplotlib.rcParams["ytick.labelsize"] = 10

aligned_coords = pd.read_csv("./out/aligned_coords_slideseq.csv", index_col=0).values
view_idx = pd.read_csv("./out/view_idx_slideseq.csv", index_col=0).values
X = pd.read_csv("./out/X_slideseq.csv", index_col=0).values
Y = pd.read_csv("./out/Y_slideseq.csv", index_col=0).values
data = sc.read_h5ad("./out/data_slideseq.h5")

landmark_markersize = 600
plt.style.use("dark_background")

view_idx = []
for vv in range(2):
    view_idx.append(np.where(data.obs.batch.values == str(vv))[0])


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

# view1_landmark_locs_postalignment = np.array(
#     [
#         [2.63, -0.28],
#         [-0.81, 2.22],
#     ]
# )
# view2_landmark_locs_postalignment = np.array(
#     [
#         [1.89, 0.24],
#         [-0.97, 3.22],
#     ]
# )


# Get spots closest to landmarks so we can track them post-alignment
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


plt.figure(figsize=(24, 7))

# colors = ["magenta", "cyan"]
colors = ["blue", "orange"]
plt.subplot(131)
plt.title("Original data")
for vv in range(len(data.obs.batch.unique())):
    plt.scatter(
        X[view_idx[vv], 0],
        X[view_idx[vv], 1],
        # s=1,
        label="View {}".format(vv + 1),
        c=colors[vv],
        alpha=0.5,
        # aligned_coords[view_idx[vv], 0], aligned_coords[view_idx[vv], 1], s=1, label="View {}".format(vv + 1)
    )

for ll in range(len(view1_landmark_locs_prealignment)):
    plt.scatter(
        X[view_idx[0]][close_idx_1[ll], 0],
        X[view_idx[0]][close_idx_1[ll], 1],
        color="yellow",
        marker="*",
        s=landmark_markersize,
    )
    # plt.scatter(
    #     X[view_idx[0]][close_idx_1[1], 0],
    #     X[view_idx[0]][close_idx_1[1], 1],
    #     color="red",
    #     marker="o",
    # )

    plt.scatter(
        X[view_idx[1]][close_idx_2[ll], 0],
        X[view_idx[1]][close_idx_2[ll], 1],
        color="lime",
        marker="*",
        s=landmark_markersize,
    )
    # plt.scatter(
    #     X[view_idx[1]][close_idx_2[1], 0],
    #     X[view_idx[1]][close_idx_2[1], 1],
    #     color="green",
    #     marker="o",
    # )
plt.gca().invert_yaxis()
plt.axis("off")

plt.subplot(132)
plt.title("Aligned")
for vv in range(len(data.obs.batch.unique())):
    plt.scatter(
        # X[view_idx[vv], 0], X[view_idx[vv], 1], s=1, label="View {}".format(vv + 1)
        aligned_coords[view_idx[vv], 0],
        aligned_coords[view_idx[vv], 1],
        # s=1,
        label="View {}".format(vv + 1),
        c=colors[vv],
        alpha=0.5,
    )

for ll in range(len(view1_landmark_locs_prealignment)):
    plt.scatter(
        aligned_coords[view_idx[0]][close_idx_1[ll], 0],
        aligned_coords[view_idx[0]][close_idx_1[ll], 1],
        color="yellow",
        marker="*",
        s=landmark_markersize,
    )
    # plt.scatter(
    #     aligned_coords[view_idx[0]][close_idx_1[1], 0],
    #     aligned_coords[view_idx[0]][close_idx_1[1], 1],
    #     color="red",
    #     marker="o",
    # )

    plt.scatter(
        aligned_coords[view_idx[1]][close_idx_2[ll], 0],
        aligned_coords[view_idx[1]][close_idx_2[ll], 1],
        color="lime",
        marker="*",
        s=landmark_markersize,
    )
    # plt.scatter(
    #     aligned_coords[view_idx[1]][close_idx_2[1], 0],
    #     aligned_coords[view_idx[1]][close_idx_2[1], 1],
    #     color="green",
    #     marker="o",
    # )
plt.gca().invert_yaxis()
plt.axis("off")

plt.subplot(133)
plt.title("Deformation field")

grid_size = 20
neighbor_dist_threshold = 1

view_idx_to_plot = 1
X_unaligned = X[view_idx[view_idx_to_plot]].copy()
X_aligned = aligned_coords[view_idx[view_idx_to_plot]].copy()
assert len(X_unaligned) == len(X_aligned)

x1s = np.linspace(X_unaligned[:, 0].min(), X_unaligned[:, 0].max(), num=grid_size)
x2s = np.linspace(X_unaligned[:, 1].min(), X_unaligned[:, 1].max(), num=grid_size)
X1, X2 = np.meshgrid(x1s, x2s)

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


# plt.scatter(grid_points[:, 0], grid_points[:, 1], marker="+", color="gray")
deformation_grid_x = np.zeros(X1.shape)
deformation_grid_y = np.zeros(X2.shape)


# for ii, gp in enumerate(grid_points):
for ii in range(grid_size):
    for jj in range(grid_size):
        dists = pairwise_distances(np.array([X1[ii, jj], X2[ii, jj]]).reshape(1, -1), X_unaligned).squeeze()
        curr_neighbor_idx = np.where(dists < neighbor_dist_threshold)[0]
        if len(curr_neighbor_idx) == 0:
            avg_displacement = [0, 0]
        else:
            avg_displacement = (X_aligned[curr_neighbor_idx] - X_unaligned[curr_neighbor_idx]).mean(0)
        deformation_grid_x[ii, jj] = X1[ii, jj] + avg_displacement[0]
        deformation_grid_y[ii, jj] = X2[ii, jj] + avg_displacement[1]

        if len(curr_neighbor_idx) != 0:
            plt.arrow(X1[ii, jj], X2[ii, jj], avg_displacement[0], avg_displacement[1], head_width=0.2)
plt.gca().invert_yaxis()
plt.axis("off")
plt.savefig("./out/landmark_dists_slideseq_scatter.png")
plt.show()
plt.close()

import ipdb

ipdb.set_trace()


## Bar plot showing change in distance between landmarks
prealignment_dists = np.sum(
    (X[view_idx[0]][close_idx_1] - X[view_idx[1]][close_idx_2]) ** 2, axis=1
)
postalignment_dists = np.sum(
    (
        aligned_coords[view_idx[0]][close_idx_1]
        - aligned_coords[view_idx[1]][close_idx_2]
    )
    ** 2,
    axis=1,
)

plt.figure(figsize=(10, 5))
dists_df = pd.melt(
    pd.DataFrame(
        {"Pre-alignment": prealignment_dists, "Post-alignment": postalignment_dists},
        index=["Landmark 1", "Landmark 2", "Landmark 3"],
    ),
    ignore_index=False,
)
dists_df["Landmark"] = dists_df.index.values
sns.barplot(data=dists_df, x="Landmark", y="value", hue="variable")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("")
plt.ylabel("Distance")
plt.xticks(rotation=300)
plt.tight_layout()
plt.savefig("./out/landmark_dists_slideseq.png")
plt.show()

import ipdb

ipdb.set_trace()
