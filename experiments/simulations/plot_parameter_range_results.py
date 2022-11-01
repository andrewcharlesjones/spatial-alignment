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

spatial_variance_errors_df = pd.read_csv(
    "./out/error_experiment_parameter_range_spatial_variance.csv", index_col=0
)
lengthscale_errors_df = pd.read_csv(
    "./out/error_experiment_parameter_range_lengthscale.csv", index_col=0
)

# keep_idx = np.delete(np.arange(len(spatial_variance_errors_df)), 18)
# spatial_variance_errors_df = spatial_variance_errors_df.iloc[keep_idx]
# lengthscale_errors_df = lengthscale_errors_df.iloc[keep_idx]

plt.figure(figsize=(17, 6))

spatial_variance_errors_df = spatial_variance_errors_df[
    spatial_variance_errors_df.value < 1
]

## Spatial variance
plt.subplot(121)
plt.title("Spatial variance")
# sns.lineplot(data=spatial_variance_errors_df, x="variable", y="value")
sns.lineplot(data=spatial_variance_errors_df, x="variable", y="value", ci="sd")
true_spatial_variance = np.median(spatial_variance_errors_df.variable.unique())
# plt.axvline(true_spatial_variance, color="black", linestyle="--")
plt.axvline(true_spatial_variance, color="black", linestyle="--")
plt.xlabel(r"$\sigma^2$")
plt.ylabel("Error")
# plt.show()

## Length scale
plt.subplot(122)
plt.title("Length scale")
sns.lineplot(data=lengthscale_errors_df, x="variable", y="value")
true_lengthscale = np.median(lengthscale_errors_df.variable.unique())
plt.axvline(
    true_lengthscale, color="black", linestyle="--", label="Data-generating\nvalue"
)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=20)
plt.xlabel(r"$\ell$")
plt.ylabel("Error")


plt.tight_layout()
plt.savefig("./out/error_experiment_parameter_range.png")
plt.show()
plt.close()

import ipdb

ipdb.set_trace()
