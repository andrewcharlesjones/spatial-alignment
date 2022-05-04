import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import multivariate_normal as mvn

import seaborn as sns
import sys

sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data
import os
from os.path import join as pjoin
import anndata

import matplotlib

font = {"size": 25}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 35

n_spatial_dims = 2
n_views = 2
markers = ["o", "X"]

lengthscale_list = [10 ** x for x in [-1, 0, 1]]
amplitude_list = [0.1, 1.0, 5.0]

xlimits = [0, 10]
ylimits = [0, 10]
grid_size = 10
x1s = np.linspace(*xlimits, num=grid_size)
x2s = np.linspace(*ylimits, num=grid_size)
X1, X2 = np.meshgrid(x1s, x2s)
X = np.vstack([X1.ravel(), X2.ravel()]).T
n = len(X)

## Lengthscales
plt.figure(figsize=(17, 15))
for ii, lengthscale in enumerate(lengthscale_list):
    for jj, amplitude in enumerate(amplitude_list):
        kernel = RBF(length_scale=lengthscale)
        K = amplitude * kernel(X) + 1e-8 * np.eye(n)
        X_warped = np.zeros((n, n_spatial_dims))
        for dd in range(n_spatial_dims):
            X_warped[:, dd] = mvn.rvs(mean=X[:, dd], cov=K)

        plt.subplot(
            len(amplitude_list),
            len(lengthscale_list),
            ii * len(lengthscale_list) + jj + 1,
        )
        plt.scatter(
            X[:, 0],
            X[:, 1],
            color="gray",
            marker=markers[0],
            label="Original" if (jj == len(amplitude_list) - 1) and (ii == 0) else None,
            s=50,
        )
        plt.scatter(
            X_warped[:, 0],
            X_warped[:, 1],
            color="red",
            marker=markers[1],
            label="Warped" if (jj == len(amplitude_list) - 1) and (ii == 0) else None,
            s=50,
        )
        plt.title(
            r"$\ell^2$ = "
            + str(round(lengthscale, 2))
            + ", $\sigma^2$ = "
            + str(round(amplitude, 2))
        )

        if (jj == len(amplitude_list) - 1) and (ii == 0):
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.xticks([])
        plt.yticks([])
plt.tight_layout()
plt.savefig("./out/warp_parameter_demo.png")
plt.show()
import ipdb

ipdb.set_trace()
