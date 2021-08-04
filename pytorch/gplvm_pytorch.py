import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.contrib.gp as gp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util import make_pinwheel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def RBF(x, y, lengthscale, variance, jitter=None):
    N = x.size()[0]
    x = x / lengthscale
    y = y / lengthscale
    s_x = torch.sum(torch.pow(x, 2), dim=1).reshape([-1, 1])
    s_y = torch.sum(torch.pow(y, 2), dim=1).reshape([1, -1])
    K = variance * torch.exp(-0.5 * (s_x + s_y - 2 * torch.mm(x, y.t())))
    if jitter is not None:
        K += jitter * torch.eye(N)
    return K


# Define model
class GPLVM(nn.Module):
    def __init__(self, n, p):
        super(GPLVM, self).__init__()
        self.X = nn.Parameter(torch.randn(size=[n, p]) * 0.1, requires_grad=True)
        self.noise_variance = nn.Parameter(
            torch.randn(size=[1]) * 0.1, requires_grad=True
        )
        self.kernel_variance = nn.Parameter(
            torch.randn(size=[1]) * 0.1, requires_grad=True
        )
        self.lengthscale = nn.Parameter(torch.randn(size=[1]) * 0.1, requires_grad=True)
        self.n = n
        self.p = p

    def forward(self):
        self.noise_variance_pos = torch.exp(self.noise_variance) + 1e-4
        self.kernel_variance_pos = torch.exp(self.kernel_variance)
        self.lengthscale_pos = torch.exp(self.lengthscale)
        # Kxx = RBF(self.X, self.X, variance=self.kernel_variance_pos, lengthscale=self.lengthscale_pos)
        self.kernel = lambda x, y: RBF(
            x, y, variance=self.kernel_variance_pos, lengthscale=self.lengthscale_pos
        )
        Kxx = self.kernel(self.X, self.X)
        return self.X, Kxx + self.noise_variance_pos * torch.eye(self.n)


class GPLVMDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, Y):
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.Y[idx, :]


if __name__ == "__main__":

    Y = make_pinwheel(
        radial_std=0.3, tangential_std=0.05, num_classes=3, num_per_class=30, rate=0.4
    )
    n, p = Y.shape

    model = GPLVM(n, p).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = GPLVMDataset(Y.astype("float64"))
    train_dataloader = DataLoader(
        dataset, batch_size=dataset.__len__(), shuffle=False, num_workers=0
    )

    def loss_fn(X, cov_mat, data):
        loss = 0.0

        # Likelihood
        output_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(n), covariance_matrix=cov_mat
        )
        LL = torch.sum(
            torch.stack(
                [
                    output_dist.log_prob(data[:, ii].float())
                    for ii in range(data.shape[1])
                ]
            )
        )

        # Prior
        prior_dist = torch.distributions.Normal(loc=0, scale=1)
        prior = torch.sum(prior_dist.log_prob(X))

        loss = -LL - prior
        return loss

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, y in enumerate(dataloader):
            y = y.to(device)

            # Compute prediction error
            pred_X, pred_cov = model()
            loss = loss_fn(pred_X, pred_cov, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    latent_ax = fig.add_subplot(121, frameon=False)
    data_ax = fig.add_subplot(122, frameon=False)
    plt.show(block=False)

    def callback(data, latent_vars):

        data_ax.cla()
        data_ax.plot(data[:, 0], data[:, 1], "bx")
        data_ax.set_xticks([])
        data_ax.set_yticks([])
        data_ax.set_title("Observed Data")

        latent_ax.cla()
        latent_ax.plot(latent_vars[:, 0], latent_vars[:, 1], "kx")
        latent_ax.set_xticks([])
        latent_ax.set_yticks([])
        latent_ax.set_title("Latent coordinates")

        plt.draw()
        plt.pause(1.0 / 60.0)

    epochs = 2000
    for t in range(epochs):
        loss = train(train_dataloader, model, loss_fn, optimizer)
        if t % 100 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"loss: {loss:>7f}")
            callback(Y, model.X.detach().numpy())
        # test(test_dataloader, model, loss_fn)
    print("Done!")

    import ipdb

    ipdb.set_trace()
