import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import pandas as pd

from gpsa import VariationalGPSA
from gpsa import matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod
import sys

sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

device = "cuda" if torch.cuda.is_available() else "cpu"

results_df = pd.read_csv("out/error_experiment_large_numspots.csv", index_col=0)
# results_df = results_df[results_df.value < 0.8]
# import ipdb; ipdb.set_trace()

plt.figure(figsize=(7, 6))
sns.boxplot(data=results_df, x="method", y="value")
plt.xlabel("")
plt.ylabel("Error")

plt.tight_layout()
plt.savefig("./out/error_experiment_large_numspots.png")
plt.show()
plt.close()

import ipdb

ipdb.set_trace()
