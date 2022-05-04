import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys

from gpsa import VariationalGPSA, rbf_kernel
from gpsa.plotting import callback_twod

sys.path.append("../../..")
sys.path.append("../../../data/st")
from load_st_data import load_st_data

sys.path.append("../../../data")
from warps import apply_gp_warp, apply_linear_warp, apply_polar_warp


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

## For PASTE
import scanpy as sc
import squidpy as sq
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../paste")
from src.paste import PASTE, visualization

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


results_df = pd.read_csv("./out/st_alignment_synthetic_warp_numgenes.csv", index_col=0)


plt.figure(figsize=(7, 7))
sns.boxplot(data=results_df, x="Type", y="value")
# plt.xscale("log")
plt.xlabel("Gene subset")
plt.ylabel("Error")
plt.xticks(rotation=45)
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./out/st_alignment_synthetic_warp_numgenes.png")
plt.show()

import ipdb

ipdb.set_trace()
