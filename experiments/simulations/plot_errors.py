import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


error_df_n_outputs = pd.read_csv("./out/error_vary_n_outputs.csv", index_col=0)
error_df_warp_magnitude = pd.read_csv(
    "./out/error_vary_warp_magnitude.csv", index_col=0
)
error_df_noise_variance = pd.read_csv(
    "./out/error_vary_noise_variance.csv", index_col=0
)


# import ipdb; ipdb.set_trace()
plt.figure(figsize=(20, 5))

plt.subplot(131)
g = sns.lineplot(
    data=error_df_n_outputs, x="variable", y="value", hue="method", err_style="bars"
)

# plt.legend(
#     bbox_to_anchor=(1.1, 1.05), loc=2, borderaxespad=0.,
# )
# g.legend_.set_title(None)
g.legend_.remove()

plt.xlabel("Number of outputs")
plt.ylabel("Error")

plt.subplot(132)
g = sns.lineplot(
    data=error_df_warp_magnitude,
    x="variable",
    y="value",
    hue="method",
    err_style="bars",
)

# plt.legend(
#     bbox_to_anchor=(1.1, 1.05), loc=2, borderaxespad=0.,
# )
# g.legend_.set_title(None)
g.legend_.remove()

plt.xlabel("Magnitude of distortion")
plt.ylabel("Error")
plt.tight_layout()

plt.subplot(133)
g = sns.lineplot(
    data=error_df_noise_variance,
    x="variable",
    y="value",
    hue="method",
    err_style="bars",
)

plt.legend(
    bbox_to_anchor=(1.1, 1.05),
    loc=2,
    borderaxespad=0.0,
)
g.legend_.set_title(None)

plt.xlabel("Noise variance")
plt.ylabel("Error")
plt.tight_layout()


plt.savefig("../../plots/error_plot.png")
plt.show()

import ipdb

ipdb.set_trace()
