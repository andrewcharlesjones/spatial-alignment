import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib

font = {"size": 15}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True


data_path = "./out/intermediate_st_X.csv"
G_path = "./out/intermediate_st_G.csv"
layer2_path = "./data/st/layer2.csv"
expression = pd.read_csv(layer2_path, index_col=0)

## Select high-variance genes
n_genes = 10
gene_vars = expression.var(0).values
sorted_idx = np.argsort(-gene_vars)
chosen_idx = sorted_idx[:n_genes]
gene_names = expression.columns.values[chosen_idx]
Y_orig_unstdized = np.log(expression.values[:, chosen_idx] + 1)
Y = np.concatenate([Y_orig_unstdized, Y_orig_unstdized], axis=0)

X = pd.read_csv(data_path).values
G = pd.read_csv(G_path).values

n = X.shape[0]
view_idx = [np.arange(0, n // 2), np.arange(n // 2, n)]

plt.figure(figsize=(14, 7))
markers = [".", "+", "^"]

plt.subplot(121)
for ii, curr_view_idx in enumerate(view_idx):
    curr_X = X[curr_view_idx]
    plt.scatter(
        curr_X[:, 0],
        curr_X[:, 1],
        label="View {}".format(ii + 1),
        marker=markers[ii],
        s=150,
        c=Y[curr_view_idx, 0],
    )
    plt.legend(loc="upper right")
    plt.xlabel("Spatial dimension 1")
    plt.ylabel("Spatial dimension 2")
    plt.title(r"Observed data space $\mathbf{X}$")
plt.colorbar()

plt.subplot(122)
for ii, curr_view_idx in enumerate(view_idx):
    curr_X_warped = G[curr_view_idx]
    plt.scatter(
        curr_X_warped[:, 0],
        curr_X_warped[:, 1],
        label="View {}".format(ii + 1),
        marker=markers[ii],
        s=150,
        c=Y[curr_view_idx, 0],
    )
    plt.legend(loc="upper right")
    plt.xlabel("Spatial dimension 1")
    plt.ylabel("Spatial dimension 2")
    plt.title(r"Reference space $\mathbf{G}$")
plt.colorbar()

plt.tight_layout()
# plt.savefig("./plots/example_alignment_simulated.png")
plt.show()

import ipdb

ipdb.set_trace()
