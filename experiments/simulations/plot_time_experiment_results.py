import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plottify import autosize

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

results = pd.read_csv("./out/time_experiment_results.csv")
results["value"] /= 60.0

plt.figure(figsize=(10, 5))
g = sns.lineplot(data=results, x="variable", y="value", hue="method")
g.legend_.set_title(None)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.xlabel("Number of samples")
plt.ylabel("Time (mins.)")
autosize()
plt.savefig("./out/time_experiment_plot.png")
plt.show()
# import ipdb; ipdb.set_trace()
