import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.contrib.gp as gp


def gpr(X, Y):
    pass


if __name__ == "__main__":

    n = 100
    p = 1
    noise_variance = 0.1
    X_distribution = torch.distributions.Uniform(low=-10, high=10)
    X = X_distribution.sample(sample_shape=torch.Size([n, p]))

    kernel = gp.kernels.RBF(
        input_dim=p, variance=torch.tensor(1.0), lengthscale=torch.tensor(3.0)
    )
    Kxx = kernel(X, X)
    Kxx_plus_noise = Kxx + noise_variance * torch.eye(n)

    Y_distribution = torch.distributions.MultivariateNormal(
        loc=torch.zeros(n), covariance_matrix=Kxx_plus_noise
    )
    Y = Y_distribution.sample()

    ntest = 1000
    Xtest = torch.linspace(-10, 10, ntest)
    Kxtestx = kernel(Xtest, X)
    Kxx_inv = torch.linalg.inv(Kxx + noise_variance * torch.eye(n))
    Ypred = torch.matmul(torch.matmul(Kxtestx, Kxx_inv), Y)

    plt.scatter(X.numpy(), Y.numpy())
    plt.plot(Xtest.detach().numpy(), Ypred.detach().numpy(), color="red")
    plt.show()
    import ipdb

    ipdb.set_trace()
    gpr(X, Y)
