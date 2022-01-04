import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from plottify import autosize

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

fixed_view_idx = 1
n_views = 4
gene_idx = 0

aligned_coords = pd.read_csv("./out/aligned_coords_st_3d.csv", index_col=0).values
X = pd.read_csv("./out/X_st_3d.csv", index_col=0).values
Y = pd.read_csv("./out/Y_st_3d.csv", index_col=0).values
data = sc.read_h5ad("./out/data_st_3d.h5")

view_idx = []
for vv in data.obs.batch.unique():
    view_idx.append(np.where(data.obs.batch.values == vv)[0])


plt.figure(figsize=(12, 10))
for vv in range(len(view_idx)):
	curr_X = aligned_coords[view_idx[vv], 0]
	curr_Y = aligned_coords[view_idx[vv], 1]
	curr_Z = aligned_coords[view_idx[vv], 2]

	plt.subplot(221)
	plt.scatter(curr_X, curr_Z)
	plt.xlabel("X")
	plt.ylabel("Z")
	plt.yticks(np.arange(4))
	plt.title("Top view")
	plt.subplot(222)
	plt.scatter(curr_Z, curr_Y, label="Slice {}".format(vv + 1))
	plt.xlabel("Z")
	plt.ylabel("Y")
	plt.xticks(np.arange(4))
	plt.title("Side view")



aligned_coords_round2 = pd.read_csv("./out/aligned_coords_st_3d_round2.csv", index_col=0).values
X_round2 = pd.read_csv("./out/X_st_3d_round2.csv", index_col=0).values
Y_round2 = pd.read_csv("./out/Y_st_3d_round2.csv", index_col=0).values
data_round2 = sc.read_h5ad("./out/data_st_3d_round2.h5")

assert data.shape[0] == data_round2.shape[0]
# assert np.array_equal(aligned_coords[view_idx[0]], aligned_coords_round2[view_idx[0]])
assert np.array_equal(X_round2, aligned_coords)
import ipdb; ipdb.set_trace()

for vv in range(len(view_idx)):
	curr_X = aligned_coords_round2[view_idx[vv], 0]
	curr_Y = aligned_coords_round2[view_idx[vv], 1]
	curr_Z = aligned_coords_round2[view_idx[vv], 2]

	plt.subplot(223)
	plt.scatter(curr_X, curr_Z)
	plt.xlabel("X")
	plt.ylabel("Z")
	plt.yticks(np.arange(4))
	plt.title("Top view")
	plt.subplot(224)
	plt.scatter(curr_Z, curr_Y, label="Slice {}".format(vv + 1))
	plt.xlabel("Z")
	plt.ylabel("Y")
	plt.xticks(np.arange(4))
	plt.title("Side view")




plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./out/st_alignment_3d.png")
plt.show()

import ipdb; ipdb.set_trace()


