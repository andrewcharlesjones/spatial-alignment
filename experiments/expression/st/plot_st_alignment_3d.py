import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from plottify import autosize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

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

import ipdb

ipdb.set_trace()
view_idx = []
for vv in data.obs.batch.unique():
    view_idx.append(np.where(data.obs.batch.values == vv)[0])


aligned_coords_round2 = pd.read_csv(
    "./out/aligned_coords_st_3d_round2.csv", index_col=0
).values
X_round2 = pd.read_csv("./out/X_st_3d_round2.csv", index_col=0).values
Y_round2 = pd.read_csv("./out/Y_st_3d_round2.csv", index_col=0).values
data_round2 = sc.read_h5ad("./out/data_st_3d_round2.h5")
G_test = pd.read_csv("./out/G_test_round2.csv", index_col=0).values
F_pred = pd.read_csv("./out/F_pred_round2.csv", index_col=0).values

assert data.shape[0] == data_round2.shape[0]

for vv in [0, 2, 3]:
    assert np.array_equal(
        aligned_coords[view_idx[0]], aligned_coords_round2[view_idx[0]]
    )
assert np.array_equal(X_round2, aligned_coords)


# fig = plt.figure()
# ax = Axes3D(fig)
# def init():
# 	ax.scatter3D(G_test[:, 0], G_test[:, 1], G_test[:, 2], c=F_pred[:, 0])
# 	ax.set_xlabel('x')
# 	ax.set_ylabel('y')
# 	ax.set_zlabel('z')
# 	ax.set_title("Densely imputed expression\n(ST, breast cancer)")
# 	autosize()
# 	return fig,

# def animate(i):
# 	ax.view_init(elev=10., azim=i)
# 	return fig,

# # Animate
# anim = animation.FuncAnimation(fig, animate, init_func=init,
# 							   frames=360, interval=20, blit=True)
# # Save
# anim.save('./out/st_imputation_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# import ipdb; ipdb.set_trace()

plt.figure(figsize=(7, 7))
ax = plt.axes(projection="3d")
ax.scatter3D(G_test[:, 0], G_test[:, 1], G_test[:, 2], c=F_pred[:, 0])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Densely imputed expression")
plt.savefig("./out/st_imputation_3d_cube.png")
autosize()
# plt.show()
plt.close()
# import ipdb; ipdb.set_trace()


plt.figure(figsize=(21, 3))
n_z_slices = len(np.unique(G_test[:, 2]))
x3s = np.linspace(*[0, 3], num=n_z_slices)
for zz in range(n_z_slices):
    # plt.subplot(np.sqrt(n_z_slices).astype(int), np.sqrt(n_z_slices).astype(int), zz + 1)
    plt.subplot(1, n_z_slices, zz + 1)
    plt.gca().cla()

    curr_idx = np.where(G_test.squeeze()[:, 2] == x3s[zz])[0]
    plt.scatter(
        G_test.squeeze()[curr_idx, 0],
        G_test.squeeze()[curr_idx, 1],
        c=F_pred.squeeze()[curr_idx, 0],
        marker="s",
        s=100,
    )
    plt.title("Z={}".format(x3s[zz]))
    plt.xlabel("x" if zz == 0 else None)
    plt.ylabel("y" if zz == 0 else None)
    plt.axis("off")
plt.tight_layout()

# autosize()
plt.savefig("./out/st_imputation_z_axis.png")
plt.show()

import ipdb

ipdb.set_trace()


plt.figure(figsize=(12, 5))
for vv in range(len(view_idx)):
    curr_X = aligned_coords_round2[view_idx[vv], 0]
    curr_Y = aligned_coords_round2[view_idx[vv], 1]
    curr_Z = aligned_coords_round2[view_idx[vv], 2]

    plt.subplot(121)
    plt.scatter(curr_X, curr_Z)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.yticks(np.arange(4))
    # plt.title("Top view")
    plt.subplot(122)
    plt.scatter(curr_Y, curr_Z, label="Slice {}".format(vv + 1))
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.yticks(np.arange(4))
    # plt.title("Side view")

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./out/st_alignment_3d.png")
plt.show()
plt.close()

# import ipdb; ipdb.set_trace()
