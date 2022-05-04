import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
    pd.concat([errors_gpsa.mean(1), errors_union.mean(1)]), columns=["value"]
)
results_df_trialwise_mean["method"] = np.concatenate(
    [["GPSA"] * len(errors_gpsa), ["Union"] * len(errors_union)]
)
g = sns.boxplot(data=results_df_trialwise_mean, x="method", y="value", color="gray")
plt.xlabel("")
plt.ylabel(r"$R^2$")
plt.suptitle("Visium prediction")


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

plt.xlabel(r"$R^2$, Union")
plt.ylabel(r"$R^2$, GPSA")

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

plt.savefig("./out/two_d_prediction_comparison_visium.png")
plt.show()
import ipdb

ipdb.set_trace()

import ipdb

ipdb.set_trace()
