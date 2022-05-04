import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


results_df = pd.read_csv(
    "./out/st_alignment_synthetic_warp_mulitple_types.csv", index_col=0
)
# results_df =
results_df.value = np.log(results_df.value)

plt.figure(figsize=(10, 5))
sns.boxplot(data=results_df, x="Warp type", y="value", hue="variable")
plt.ylabel("log(error)")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("./out/st_alignment_synthetic_warp_mulitple_types.png")
plt.show()
plt.close()

import ipdb

ipdb.set_trace()
