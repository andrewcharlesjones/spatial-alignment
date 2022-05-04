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

n_gene_sets_to_name = 2

results_df = pd.read_csv("./out/st_avg_gene_variance_gsea_results.csv", index_col=0)
results_df["logpval"] = -np.log10(results_df.padj.values)

plt.figure(figsize=(7, 7))
sns.scatterplot(data=results_df, x="NES", y="logpval", color="black", edgecolor=None)
plt.xlabel("Enrichment score")
plt.ylabel(r"$-\log_{10}$(p-val)")

sorted_idx = np.argsort(-results_df.NES.values)
for ii in range(n_gene_sets_to_name):
    gs_name = " ".join(results_df.pathway.values[sorted_idx[ii]].split("_")[1:])
    plt.text(
        s=gs_name,
        x=results_df.NES.values[sorted_idx[ii]],
        y=results_df.logpval.values[sorted_idx[ii]],
        ha="right",
    )

plt.tight_layout()
plt.savefig("./out/st_avg_gene_variance_gsea_results.png")
plt.show()
import ipdb

ipdb.set_trace()
