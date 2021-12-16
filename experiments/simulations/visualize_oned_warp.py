import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process.kernels import RBF

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


lengthscale = 1.0
amplitude = 1.0
noise_stddev = 1e-6

xlims = [-5, 5]
n = 100
X = np.linspace(xlims[0], xlims[1], n)
X = np.expand_dims(X, 1)

## Draw function
K_XX = amplitude * RBF(length_scale=lengthscale)(X, X) + noise_stddev * np.eye(n)
mean = X.squeeze()
# mean = np.zeros(n)
Y = mvn(mean, K_XX).rvs()

# import ipdb; ipdb.set_trace()
plt.figure(figsize=(7, 6))
plt.plot(X, Y, linewidth=5)
plt.xlabel("Observed spatial coordinate")
plt.ylabel("Warped spatial coordinate")
plt.title(r"$\sigma^2 = {}, \ell = {}$".format(amplitude, lengthscale))
plt.tight_layout()
plt.savefig("../../plots/mean_function_example.png")
plt.show()
