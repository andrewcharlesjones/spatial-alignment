import sys

sys.path.append("..")
from gpr import GPRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.gaussian_process.kernels as skernels
from scipy.stats import multivariate_normal as mvn


def test_1d():
    # Toy dataset drawn from GP prior with RBF kernel
    kern = skernels.RBF()
    n = 20
    X = np.random.uniform(low=-3, high=3, size=(n, 1))
    Y = mvn.rvs(mean=np.zeros(n), cov=kern(X, X))

    # GPR
    gpr = GPRegressor(kernel=kern, use_inducing_points=False)
    gpr.fit(X, Y)

    # Plot
    Xtest = np.expand_dims(np.linspace(-3, 3, 200), 1)
    mean_pred, cov_pred = gpr.predict(Xtest)
    gpr.plot_predictions(Xtest, mean_pred, cov_pred)


def test_2d():
    # Toy dataset drawn from GP prior with RBF kernel
    kern = skernels.RBF()
    n = 20
    x1, x2 = np.random.uniform(low=-3, high=3, size=(n, 1)), np.random.uniform(
        low=-3, high=3, size=(n, 1)
    )
    X = np.hstack([x1, x2])
    Y = mvn.rvs(mean=np.zeros(n), cov=kern(X, X))

    # GPR
    gpr = GPRegressor(kernel=kern, use_inducing_points=True)
    gpr.fit(X, Y)

    # Plot
    x1, x2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Xtest = np.vstack([x1.ravel(), x2.ravel()]).T
    mean_pred, cov_pred = gpr.predict(Xtest)
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=mean_pred)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()


if __name__ == "__main__":
    test_1d()
    test_2d()
