import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


# error_df = pd.read_csv("./out/prediction_comparison_visium.csv", index_col=0)
error_df = pd.read_csv("./out/twod_prediction_visium.csv", index_col=0)

error_df = error_df[error_df["variable"] != "Separate"]


plt.figure(figsize=(7, 5))

g = sns.boxplot(data=error_df, x="variable", y="value", color="gray")
plt.xlabel("")
plt.ylabel("MSE")
plt.title("Visium prediction")
plt.tight_layout()

plt.savefig("./out/two_d_prediction_comparison_visium.png")
plt.show()

import ipdb

ipdb.set_trace()
