import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util import make_pinwheel
import seaborn as sns
from gp_functions import rbf_covariance
from scipy.stats import multivariate_normal as mvnpy
from util import polar_warp


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))


def RBF(x, y, lengthscale, variance):
    N = x.size()[0]
    x = x / lengthscale
    y = y / lengthscale
    s_x = torch.sum(torch.pow(x, 2), dim=1).reshape([-1, 1])
    s_y = torch.sum(torch.pow(y, 2), dim=1).reshape([1, -1])
    K = variance * torch.exp(-0.5 * (s_x + s_y - 2 * torch.mm(x, y.t())))
    return K


# Define model
class TwoLayerWarpGP(nn.Module):
    def __init__(
        self,
        n_views,
        n_samples_list,
        n_features,
        G_init=None,
        n_spatial_dims=2,
        n_kernel_params=4,
        n_noise_variance_params=1,
    ):
        super(TwoLayerWarpGP, self).__init__()
        self.N = np.sum(n_samples_list)
        self.D = n_spatial_dims
        self.P = n_features
        self.n_views = n_views
        self.n_samples_list = n_samples_list
        self.n_spatial_dims = n_spatial_dims
        self.n_kernel_params = n_kernel_params
        self.n_noise_variance_params = n_noise_variance_params
        cumulative_sums = np.cumsum(n_samples_list)
        cumulative_sums = np.insert(cumulative_sums, 0, 0)
        self.view_idx = np.array(
            [
                np.arange(cumulative_sums[ii], cumulative_sums[ii + 1])
                for ii in range(self.n_views)
            ]
        )
        self.info_dict = {"n_iters_trained": 0}
        self.print_every = 1

        ## Parameters
        if G_init is None:
            self.G = nn.Parameter(torch.randn([self.N, self.D]))
        else:
            self.G = nn.Parameter(G_init)
            # self.G_param = nn.Parameter(G_init[self.view_idx[0], :])
        self.noise_variance = nn.Parameter(torch.randn([self.n_noise_variance_params]))
        self.kernel_variances = nn.Parameter(torch.randn([self.n_kernel_params // 2]))
        self.kernel_lengthscales = nn.Parameter(
            torch.randn([self.n_kernel_params // 2])
        )
        self.mean_slopes = nn.Parameter(
            torch.randn([self.n_spatial_dims, self.n_spatial_dims])
        )
        self.mean_intercepts = nn.Parameter(torch.randn([self.n_spatial_dims]))
        self.diagonal_offset = 1e-3

    def forward(self, X_spatial):
        # self.G = torch.cat([self.G_param, X_spatial[self.view_idx[1]]])
        noise_variance_pos = torch.exp(self.noise_variance) + 1e-4
        kernel_variances_pos = torch.exp(self.kernel_variances)
        kernel_lengthscales_pos = torch.exp(self.kernel_lengthscales)
        kernel_G = lambda x, y: RBF(
            x,
            y,
            variance=kernel_variances_pos[0],
            lengthscale=kernel_lengthscales_pos[0],
        )
        kernel_Y = lambda x, y: RBF(
            x,
            y,
            variance=kernel_variances_pos[1],
            lengthscale=kernel_lengthscales_pos[1],
        )
        mean_G = torch.matmul(X_spatial, self.mean_slopes) + self.mean_intercepts
        cov_G_list = []
        for vv in range(self.n_views):
            curr_n = len(self.view_idx[vv])
            cov_G = kernel_G(
                X_spatial[self.view_idx[vv], :], X_spatial[self.view_idx[vv], :]
            ) + self.diagonal_offset * torch.eye(curr_n)
            cov_G_list.append(cov_G)

        mean_Y = torch.zeros(self.N)
        cov_Y = kernel_Y(self.G, self.G) + noise_variance_pos * torch.eye(self.N)

        return self.G, mean_G, cov_G_list, mean_Y, cov_Y


def distance_matrix(X, Y):
    squared_diffs = torch.square(torch.unsqueeze(X, 0) - torch.unsqueeze(Y, 1))
    squared_distances = torch.sum(squared_diffs, dim=2)
    return squared_distances


def loss_fn(X, Y, G, mean_G, cov_G_list, mean_Y, cov_Y, view_idx):
    n_views = len(view_idx)
    n_spatial_dims = G.shape[1]
    P = Y.shape[1]

    # Log likelihood of G given X (warp likelihood)
    LL_G = 0
    for vv in range(n_views):

        LL_G += torch.sum(
            torch.stack(
                [
                    torch.distributions.MultivariateNormal(
                        loc=mean_G[view_idx[vv], dd], covariance_matrix=cov_G_list[vv]
                    ).log_prob(G[view_idx[vv], dd])
                    for dd in range(n_spatial_dims)
                ]
            )
        )

    # Log likelihood of Y given G
    LL_Y = torch.sum(
        torch.stack(
            [
                torch.distributions.MultivariateNormal(
                    loc=mean_Y, covariance_matrix=cov_Y
                ).log_prob(Y[:, jj])
                for jj in range(P)
            ]
        )
    )

    # Penalty for preserving pairwise distances between points
    # distance_penalty = 0
    # for vv in range(n_views):
    # 	curr_distance_mat_X = distance_matrix(X[view_idx[vv], :], X[view_idx[vv], :])
    # 	curr_distance_mat_G = distance_matrix(G[view_idx[vv], :], G[view_idx[vv], :])
    # 	curr_penalty = torch.sum(torch.square(curr_distance_mat_X - curr_distance_mat_G))
    # 	distance_penalty += curr_penalty

    return -LL_G - LL_Y  # + 1e4 * distance_penalty


def train(dataloader, model, loss_fn, optimizer, device, view_idx):
    model.train()
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # Compute prediction error
        G, mean_G, cov_G_list, mean_Y, cov_Y = model.forward(x)
        loss = loss_fn(x, y, G, mean_G, cov_G_list, mean_Y, cov_Y, view_idx)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


class SpatialDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert X.shape[0] == Y.shape[0]

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]


if __name__ == "__main__":

    n_views = 2
    n_genes = 10
    kernel = rbf_covariance
    kernel_params_true = np.array([1.0, 1.0])
    n_samples_per_view = 30
    n_samples_list = [n_samples_per_view] * n_views
    cumulative_sums = np.cumsum(n_samples_list)
    cumulative_sums = np.insert(cumulative_sums, 0, 0)
    view_idx = np.array(
        [
            np.arange(cumulative_sums[ii], cumulative_sums[ii + 1])
            for ii in range(n_views)
        ]
    )
    n = np.sum(n_samples_list)
    sigma2 = 1
    X_orig = np.hstack(
        [
            np.random.uniform(low=-3, high=3, size=(n_samples_per_view, 1))
            for _ in range(2)
        ]
    )
    Y_orig = np.vstack(
        [
            mvnpy.rvs(
                mean=np.zeros(n_samples_per_view),
                cov=kernel(X_orig, X_orig, kernel_params_true),
            )
            for _ in range(n_genes)
        ]
    ).T

    X = np.empty((np.sum(n_samples_list), 2))
    Y = np.empty((np.sum(n_samples_list), n_genes))

    for vv in range(n_views):

        curr_X = X_orig.copy()
        # Warp
        linear_coeffs = np.random.normal(scale=0.1, size=2 * 2)
        rs_true, thetas_true = curr_X @ linear_coeffs[:2], curr_X @ linear_coeffs[2:]

        curr_X_observed = polar_warp(curr_X, rs_true, thetas_true)
        X[view_idx[vv]] = curr_X_observed

        curr_Y = Y_orig.copy()
        Y[view_idx[vv]] = curr_Y + np.random.normal(scale=0.1, size=curr_Y.shape)

    model = TwoLayerWarpGP(n_views=n_views, n_samples_list=n_samples_list).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = SpatialDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    train_dataloader = DataLoader(
        dataset, batch_size=dataset.__len__(), shuffle=False, num_workers=0
    )

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, sample in enumerate(dataloader):
            x = sample["X"].to(device)
            y = sample["Y"].to(device)

            # Compute prediction error
            G, mean_G, cov_G_list, mean_Y, cov_Y = model(x)
            loss = loss_fn(y, G, mean_G, cov_G_list, mean_Y, cov_Y, view_idx)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    data_ax = fig.add_subplot(121, frameon=False)
    latent_ax = fig.add_subplot(122, frameon=False)
    plt.show(block=False)

    def callback(model):
        markers = [".", "+", "^"]

        data_ax.cla()
        latent_ax.cla()

        for vv in range(n_views):
            data_ax.scatter(
                X[view_idx[vv], 0],
                X[view_idx[vv], 1],
                c=Y[view_idx[vv], 0],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )
            latent_ax.scatter(
                model.G.detach().numpy()[view_idx[vv], 0],
                model.G.detach().numpy()[view_idx[vv], 1],
                c=Y[view_idx[vv], 0],
                label="View {}".format(vv + 1),
                marker=markers[vv],
                s=100,
            )

        plt.draw()
        plt.pause(1.0 / 60.0)

    epochs = 2000
    for t in range(epochs):
        loss = train(train_dataloader, model, loss_fn, optimizer)

        if t % 100 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"loss: {loss:>7f}")
            print(X)
            with torch.no_grad():
                callback(model)
    print("Done!")

    import ipdb

    ipdb.set_trace()
