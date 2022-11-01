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
from matplotlib.collections import LineCollection

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

grid_size = 20
neighbor_dist_threshold = 1

view_idx_to_plot = 1
X_unaligned = X[view_idx[view_idx_to_plot]].copy()
X_aligned = aligned_coords[view_idx[view_idx_to_plot]].copy()
assert len(X_unaligned) == len(X_aligned)

x1s = np.linspace(X_unaligned[:, 0].min(), X_unaligned[:, 0].max(), num=grid_size)
x2s = np.linspace(X_unaligned[:, 1].min(), X_unaligned[:, 1].max(), num=grid_size)
X1, X2 = np.meshgrid(x1s, x2s)
# X_orig_single = np.vstack([X1.ravel(), X2.ravel()]).T
# grid_points = np.concatenate([X_orig_single.copy(), X_orig_single.copy()], axis=0)

# dists = pairwise_distances(grid_points, X_unaligned)

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

plt.figure(figsize=(7, 7))

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
            plt.arrow(X1[ii, jj], X2[ii, jj], avg_displacement[0], avg_displacement[1], head_width=0.1)
plt.gca().invert_yaxis()
plt.savefig("./out/slideseq_deformation_field.png")
plt.show()
import ipdb

ipdb.set_trace()


plt.figure(figsize=(7, 7))
plot_grid(X1, X2, color="gray")
plot_grid(deformation_grid_x, deformation_grid_y)
plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
plt.show()
import ipdb

ipdb.set_trace()

