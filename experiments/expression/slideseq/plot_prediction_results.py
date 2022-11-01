import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


# error_df = pd.read_csv("./out/prediction_comparison_visium.csv", index_col=0)
# error_df = pd.read_csv("./out/twod_prediction_visium.csv", index_col=0)

errors_union = pd.read_csv("./out/prediction_errors_union.csv", index_col=0)
errors_separate = pd.read_csv("./out/prediction_errors_separate.csv", index_col=0)
errors_gpsa = pd.read_csv("./out/prediction_errors_gpsa.csv", index_col=0)

errors_union_melted = pd.melt(errors_union)
errors_union_melted["method"] = "Union"

errors_separate_melted = pd.melt(errors_separate)
errors_separate_melted["method"] = "Separate"

errors_gpsa_melted = pd.melt(errors_gpsa)
errors_gpsa_melted["method"] = "GPSA"

results_df = pd.concat(
    [errors_union_melted, errors_separate_melted, errors_gpsa_melted], axis=0
)


results_df_means = results_df.groupby(["variable", "method"], as_index=False).mean()
results_df_stddevs = results_df.groupby(["variable", "method"], as_index=False).std()

results_df_gpsa = pd.merge(
    results_df_means[results_df_means.method == "GPSA"],
    results_df_stddevs[results_df_stddevs.method == "GPSA"],
    on=["variable", "method"],
    suffixes=["_mean", "_stddev"],
)
results_df_union = pd.merge(
    results_df_means[results_df_means.method == "Union"],
    results_df_stddevs[results_df_stddevs.method == "Union"],
    on=["variable", "method"],
    suffixes=["_mean", "_stddev"],
)
assert np.array_equal(results_df_gpsa.variable.values, results_df_union.variable.values)


plt.figure(figsize=(14, 7))

plt.subplot(121)

# results_df_trialwise_mean = results_df.groupby(["method", "variable"], as_index=False).mean()
# results_df_trialwise_mean = results_df_trialwise_mean[results_df_trialwise_mean.method != "Separate"]
results_df_trialwise_mean = pd.DataFrame(
    pd.concat([errors_union.mean(1), errors_gpsa.mean(1)]), columns=["value"]
)
results_df_trialwise_mean["method"] = np.concatenate(
    [["Union"] * len(errors_union), ["GPSA"] * len(errors_gpsa)]
)
g = sns.boxplot(data=results_df_trialwise_mean, x="method", y="value", color="gray")
plt.xlabel("")
plt.ylabel(r"Pearson $\rho$")
plt.suptitle("Slide-seqV2 prediction")


plt.subplot(122)
plt.errorbar(
    x=results_df_union.value_mean.values,
    y=results_df_gpsa.value_mean.values,
    xerr=results_df_union.value_stddev.values,
    yerr=results_df_gpsa.value_stddev.values,
    fmt="o",
    ecolor="black",
    color="black",
)

plt.xlabel(r"Pearson $\rho$, Union")
plt.ylabel(r"Pearson $\rho$, GPSA")

ax = plt.gca()

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, "k-", alpha=0.75, zorder=0, color="gray")
ax.set_aspect("equal")
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.tight_layout()

plt.savefig("./out/two_d_prediction_comparison_slideseq.png")
plt.show()



preds = pd.read_csv("./out/slideseq_preds_gpsa.csv", index_col=0)
truth = pd.read_csv("./out/slideseq_truth_gpsa.csv", index_col=0)

pearson_corrs = np.zeros(preds.shape[1])
for jj in range(preds.shape[1]):
    pearson_corrs[jj] = pearsonr(truth.iloc[:, jj].values, preds.iloc[:, jj].values)[0]

sorted_idx = np.argsort(pearson_corrs)
n_genes_to_plot = 3

plt.figure(figsize=(n_genes_to_plot * 7, 14))

gene_names = pd.read_csv("./out/slideseq_pred_gene_names.csv").iloc[:, 0].values

# import ipdb

# ipdb.set_trace()

for ii, gene_idx in enumerate(sorted_idx[-n_genes_to_plot:]):
    plt.subplot(2, n_genes_to_plot, ii + 1)
    plt.scatter(truth.iloc[:, gene_idx].values, preds.iloc[:, gene_idx].values, c="gray")
    plt.xlabel("True expression")
    plt.ylabel("Predicted expression")
    plt.title(r"$\emph{" + gene_names[gene_idx].upper() + "}$")

for ii, gene_idx in enumerate(sorted_idx[:n_genes_to_plot]):
    plt.subplot(2, n_genes_to_plot, ii + 4)
    plt.scatter(truth.iloc[:, gene_idx].values, preds.iloc[:, gene_idx].values, c="gray")
    plt.xlabel("True expression")
    plt.ylabel("Predicted expression")
    plt.title(r"$\emph{" + gene_names[gene_idx].upper() + "}$")

plt.tight_layout()
plt.savefig("./out/slideseq_prediction_examples.png")
plt.show()

# for jj in range(preds.shape[1]):
#     # import ipdb; ipdb.set_trace()
#     print(round(pearsonr(truth.iloc[:, jj].values, preds.iloc[:, jj].values)[0], 3))
#     nonzero_idx = np.where(truth.iloc[:, jj].values != np.min(truth.iloc[:, jj].values))[0]
#     print(round(pearsonr(truth.iloc[:, jj].values[nonzero_idx], preds.iloc[:, jj].values[nonzero_idx])[0], 3))
#     print()
#     plt.scatter(truth.iloc[:, jj].values, preds.iloc[:, jj].values, c="gray")
#     plt.xlabel("True expression")
#     plt.ylabel("Predicted expression")
#     plt.show()
#     # import ipdb; ipdb.set_trace()


import ipdb

ipdb.set_trace()
