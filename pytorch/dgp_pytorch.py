import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.contrib.gp as gp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append("..")
from util import make_pinwheel
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def RBF(x, y, lengthscale, variance, jitter=None, diag=False):
    N = x.size()[0]
    x = x / lengthscale
    y = y / lengthscale
    if diag:
        exponand = torch.sum(torch.pow(x - y, 2), dim=1)
        # s_y = torch.sum(torch.pow(y, 2), dim=1)
        K = variance * torch.exp(-0.5 * exponand)
    else:
        s_x = torch.sum(torch.pow(x, 2), dim=1).reshape([-1, 1])
        s_y = torch.sum(torch.pow(y, 2), dim=1).reshape([1, -1])
        K = variance * torch.exp(-0.5 * (s_x + s_y - 2 * torch.mm(x, y.t())))
    if jitter is not None:
        K += jitter * torch.eye(N)
    return K


# Define model
class VGPR(nn.Module):
    def __init__(self, X, n, p, m):
        super(VGPR, self).__init__()
        self.nugget = 1e-3

        # Pseudo-inputs (inducing locations)
        Xtilde_distribution = torch.distributions.Uniform(
            low=torch.min(X), high=torch.max(X)
        )
        Xtilde = X_distribution.sample(sample_shape=torch.Size([m, p]))
        self.Xtilde = nn.Parameter(Xtilde, requires_grad=True)
        # import ipdb; ipdb.set_trace()
        # self.Xtilde = X.clone()
        Omega_sqt = Omega_sqt = 0.01 * torch.randn(size=[m, m])
        Omega = torch.mm(Omega_sqt, torch.t(Omega_sqt))
        Omega_tril = torch.cholesky(Omega) + self.nugget * torch.eye(m)
        ind = np.diag_indices(Omega_tril.shape[0])
        # Omega_tril[ind[0], ind[1]] = torch.exp(torch.diag(Omega_tril))
        self.Omega_tril = nn.Parameter(Omega_tril)
        self.delta = nn.Parameter(0.1 * torch.randn(size=[m]))
        self.noise_variance = nn.Parameter(
            torch.randn(size=[1]) * 0.1, requires_grad=True
        )
        self.kernel_variance = nn.Parameter(
            torch.randn(size=[1]) * 0.1, requires_grad=True
        )
        self.lengthscale = nn.Parameter(torch.randn(size=[1]) * 0.1, requires_grad=True)
        self.n = n
        self.p = p
        self.m = m

    def forward(self, X, S=1):
        self.noise_variance_pos = torch.exp(self.noise_variance) + 1e-4
        self.kernel_variance_pos = torch.exp(self.kernel_variance)
        self.lengthscale_pos = torch.exp(self.lengthscale)
        self.kernel = lambda x, y, diag=False: RBF(
            x,
            y,
            variance=self.kernel_variance_pos,
            lengthscale=self.lengthscale_pos,
            diag=diag,
        )

        ind = np.diag_indices(self.Omega_tril.shape[0])
        # self.curr_Omega_tril_G = self.Omega_tril_G.clone()
        # self.curr_Omega_tril_G[ind[0], ind[1]] = torch.exp(torch.diag(self.curr_Omega_tril_G))
        # self.curr_Omega_tril_G += self.nugget * torch.eye(m)

        self.curr_Omega_tril = self.Omega_tril.clone()
        self.curr_Omega_tril[ind[0], ind[1]] = torch.exp(
            torch.diag(self.curr_Omega_tril)
        )
        self.curr_Omega_tril += self.nugget * torch.eye(m)

        ## Sample G from p(G | X, Xtilde)
        ## 		- Note that this is the distribution with
        ## 		  the pseudo-outputs marginalized out
        Kgg_diag = self.kernel(X, X, diag=True) + self.nugget
        Kuu = self.kernel(self.Xtilde, self.Xtilde) + self.nugget * torch.eye(m)
        self.Kuu_chol_G = torch.cholesky(Kuu)
        Kuf = self.kernel(self.Xtilde, X)

        # TODO: make these true mean functions (linear or something)
        mu_x = torch.zeros(X.shape[0])
        self.mu_z = torch.zeros(m)

        alpha_x = torch.cholesky_solve(Kuf, self.Kuu_chol_G)
        mu_tilde = mu_x + torch.matmul(alpha_x.t(), self.delta - self.mu_z)
        a_t_Kchol = torch.matmul(alpha_x.t(), self.Kuu_chol_G)
        aKa = torch.sum(torch.square(a_t_Kchol), dim=1)
        a_t_Omega_tril = torch.matmul(alpha_x.t(), self.curr_Omega_tril)
        aOmega_a = torch.sum(torch.square(a_t_Omega_tril), dim=1)
        Sigma_tilde = Kgg_diag - aKa + aOmega_a + self.nugget  # * torch.eye(X.shape[0])

        # Sample
        G_marginal_dist = torch.distributions.MultivariateNormal(
            loc=mu_tilde, covariance_matrix=torch.diag(Sigma_tilde)
        )
        G_samples = G_marginal_dist.rsample(sample_shape=[1])
        # import ipdb; ipdb.set_trace()

        ## Sample F from p(F | G, Gtilde)
        ## 		- Note that this is the distribution with
        ## 		  the pseudo-outputs marginalized out
        Kff_diag = self.kernel(X, X, diag=True) + self.nugget
        Kuu = self.kernel(self.Xtilde, self.Xtilde) + self.nugget * torch.eye(m)
        self.Kuu_chol = torch.cholesky(Kuu)
        Kuf = self.kernel(self.Xtilde, X)

        # TODO: make these true mean functions (linear or something)
        mu_x = torch.zeros(X.shape[0])
        self.mu_z = torch.zeros(m)

        alpha_x = torch.cholesky_solve(Kuf, self.Kuu_chol)
        mu_tilde = mu_x + torch.matmul(alpha_x.t(), self.delta - self.mu_z)
        a_t_Kchol = torch.matmul(alpha_x.t(), self.Kuu_chol)
        aKa = torch.sum(torch.square(a_t_Kchol), dim=1)
        a_t_Omega_tril = torch.matmul(alpha_x.t(), self.curr_Omega_tril)
        aOmega_a = torch.sum(torch.square(a_t_Omega_tril), dim=1)
        Sigma_tilde = Kff_diag - aKa + aOmega_a + self.nugget  # * torch.eye(X.shape[0])

        # Sample
        F_marginal_dist = torch.distributions.MultivariateNormal(
            loc=mu_tilde, covariance_matrix=torch.diag(Sigma_tilde)
        )
        F_samples = F_marginal_dist.rsample(sample_shape=[S])
        return F_samples

    def loss_fn(self, X, Y, F_samples):
        # This is the negative (approximate) ELBO

        # KL terms
        qu = torch.distributions.MultivariateNormal(
            loc=self.delta, scale_tril=torch.tril(self.curr_Omega_tril)
        )
        pu = torch.distributions.MultivariateNormal(
            loc=self.mu_z, scale_tril=self.Kuu_chol
        )
        KL_div = torch.distributions.kl.kl_divergence(qu, pu)

        # Log likelihood
        Y_distribution = torch.distributions.Normal(
            loc=F_samples, scale=self.noise_variance_pos
        )
        LL = Y_distribution.log_prob(Y)

        KL_loss = self.p * torch.mean(KL_div)
        LL_loss = -torch.sum(torch.mean(LL, dim=0))
        return LL_loss + KL_loss


class VGPRDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert X.shape[0] == Y.shape[0]

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return {"X": self.X[idx, :], "Y": self.Y[idx]}


if __name__ == "__main__":

    n = 100
    p = 1
    m = 10
    noise_variance = 0.01
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

    model = VGPR(X, n, p, m).to(device)
    F_samples = model.forward(X, S=10)
    loss = model.loss_fn(X, Y, F_samples)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    dataset = VGPRDataset(X, Y)
    train_dataloader = DataLoader(
        dataset, batch_size=dataset.__len__(), shuffle=False, num_workers=0
    )

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, sample in enumerate(dataloader):
            x = sample["X"].to(device)
            y = sample["Y"].to(device)

            # Compute prediction error
            F_samples = model(X, S=10)
            loss = loss_fn(X, Y, F_samples)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    data_ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(X, Y, pred):

        data_ax.cla()
        data_ax.scatter(X[:, 0], Y, alpha=0.3)
        data_ax.set_title("Observed Data")
        data_ax.scatter(
            model.Xtilde.detach().numpy()[:, 0], pred, color="red", label="Predictions"
        )

        data_ax.set_ylim([np.min(Y), np.max(Y)])

        plt.draw()
        plt.pause(1.0 / 60.0)

    epochs = 2000
    for t in range(epochs):
        loss = train(train_dataloader, model, model.loss_fn, optimizer)
        if t % 100 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"loss: {loss:>7f}")
            with torch.no_grad():
                F_samples_test = model.forward(model.Xtilde.detach(), S=50)
                F_samples_mean = torch.mean(F_samples_test, dim=0)
                callback(X.detach().numpy(), Y.detach().numpy(), F_samples_mean)
        # test(test_dataloader, model, loss_fn)
    print("Done!")

    import ipdb

    ipdb.set_trace()
