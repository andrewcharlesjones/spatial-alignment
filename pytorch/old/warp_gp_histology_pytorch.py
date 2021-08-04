import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy


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
class TwoLayerWarpHistologyGP(nn.Module):
    def __init__(
        self,
        n_views,
        n_samples_list_expression,
        n_samples_list_histology,
        n_features_expression,
        n_features_histology,
        G_init_expression=None,
        G_init_histology=None,
        n_spatial_dims=2,
        n_kernel_params=4,
        n_noise_variance_params=1,
    ):
        super(TwoLayerWarpHistologyGP, self).__init__()
        self.N_expression = np.sum(n_samples_list_expression)
        self.N_histology = np.sum(n_samples_list_histology)
        self.P_expression = n_features_expression
        self.P_histology = n_features_histology
        self.n_views = n_views
        self.n_samples_list_expression = n_samples_list_expression
        self.n_samples_list_histology = n_samples_list_histology
        self.n_spatial_dims = n_spatial_dims
        self.n_kernel_params = n_kernel_params
        self.n_noise_variance_params = n_noise_variance_params
        cumulative_sums = np.cumsum(n_samples_list_expression)
        cumulative_sums = np.insert(cumulative_sums, 0, 0)
        self.view_idx_expression = np.array(
            [
                np.arange(cumulative_sums[ii], cumulative_sums[ii + 1])
                for ii in range(self.n_views)
            ]
        )
        cumulative_sums = np.cumsum(n_samples_list_histology)
        cumulative_sums = np.insert(cumulative_sums, 0, 0)
        self.view_idx_histology = np.array(
            [
                np.arange(cumulative_sums[ii], cumulative_sums[ii + 1])
                for ii in range(self.n_views)
            ]
        )
        self.info_dict = {"n_iters_trained": 0}
        self.print_every = 1

        ## Parameters
        if G_init_expression is None and G_init_histology is None:
            self.G_expression = nn.Parameter(
                torch.randn([self.N_expression, self.n_spatial_dims])
            )
            self.G_histology = nn.Parameter(
                torch.randn([self.N_histology, self.n_spatial_dims])
            )
        else:
            self.G_expression = nn.Parameter(G_init_expression)
            self.G_histology = nn.Parameter(G_init_histology)
            # self.G_param = nn.Parameter(G_init[self.view_idx[0], :])
        self.noise_variance = nn.Parameter(torch.randn([self.n_noise_variance_params]))
        self.kernel_variances = nn.Parameter(torch.randn([self.n_kernel_params // 2]))
        self.kernel_lengthscales = nn.Parameter(
            torch.randn([self.n_kernel_params // 2])
        )
        self.mean_slopes = nn.Parameter(
            torch.randn([self.n_views, self.n_spatial_dims, self.n_spatial_dims])
        )
        self.mean_intercepts = nn.Parameter(
            torch.randn([self.n_views, self.n_spatial_dims])
        )
        self.diagonal_offset = 1e-3

    def forward(self, X_spatial_expression, X_spatial_histology):
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
        # import ipdb; ipdb.set_trace()
        # mean_G_expression = torch.matmul(X_spatial_expression, self.mean_slopes) + self.mean_intercepts
        # mean_G_histology = torch.matmul(X_spatial_histology, self.mean_slopes) + self.mean_intercepts
        mean_G_expression = torch.zeros([self.N_expression, self.n_spatial_dims])
        mean_G_histology = torch.zeros([self.N_histology, self.n_spatial_dims])
        cov_G_list = []
        for vv in range(self.n_views):
            curr_idx_expression = self.view_idx_expression[vv]
            curr_n_expression = len(curr_idx_expression)
            curr_idx_histology = self.view_idx_histology[vv]
            curr_n_histology = len(curr_idx_histology)
            curr_X_spatial_expression = X_spatial_expression[curr_idx_expression, :]
            curr_X_spatial_histology = X_spatial_histology[curr_idx_histology, :]
            curr_X_spatial = torch.cat(
                [curr_X_spatial_expression, curr_X_spatial_histology]
            )
            cov_G = kernel_G(
                curr_X_spatial, curr_X_spatial
            ) + self.diagonal_offset * torch.eye(curr_n_expression + curr_n_histology)

            # Compute means
            mean_G_expression[curr_idx_expression] = (
                torch.matmul(curr_X_spatial_expression, self.mean_slopes[vv])
                + self.mean_intercepts[vv]
            )
            mean_G_histology[curr_idx_histology] = (
                torch.matmul(curr_X_spatial_histology, self.mean_slopes[vv])
                + self.mean_intercepts[vv]
            )
            cov_G_list.append(cov_G)

        mean_Y_expression = torch.zeros(self.N_expression)
        mean_Y_histology = torch.zeros(self.N_histology)
        cov_Y_expression = kernel_Y(
            self.G_expression, self.G_expression
        ) + noise_variance_pos * torch.eye(self.N_expression)
        cov_Y_histology = kernel_Y(
            self.G_histology, self.G_histology
        ) + noise_variance_pos * torch.eye(self.N_histology)

        return (
            self.G_expression,
            self.G_histology,
            mean_G_expression,
            mean_G_histology,
            cov_G_list,
            mean_Y_expression,
            mean_Y_histology,
            cov_Y_expression,
            cov_Y_histology,
        )
        # return self.G_histology


def distance_matrix(X, Y):
    squared_diffs = torch.square(torch.unsqueeze(X, 0) - torch.unsqueeze(Y, 1))
    squared_distances = torch.sum(squared_diffs, dim=2)
    return squared_distances


def loss_fn(
    X_expression,
    X_histology,
    Y_expression,
    Y_histology,
    G_expression,
    G_histology,
    mean_G_expression,
    mean_G_histology,
    cov_G_list,
    mean_Y_expression,
    mean_Y_histology,
    cov_Y_expression,
    cov_Y_histology,
    view_idx_expression,
    view_idx_histology,
):
    # def loss_fn(G_histology):
    # return torch.sum(G_histology)
    n_views = len(view_idx_expression)
    n_spatial_dims = G_expression.shape[1]
    n_genes = Y_expression.shape[1]
    n_channels = Y_histology.shape[1]

    # Log likelihood of G given X (warp likelihood)
    LL_G = 0
    for vv in range(n_views):
        curr_G = torch.cat(
            [G_expression[view_idx_expression[vv]], G_histology[view_idx_histology[vv]]]
        )
        curr_mean_G = torch.cat(
            [
                mean_G_expression[view_idx_expression[vv]],
                mean_G_histology[view_idx_histology[vv]],
            ]
        )

        LL_G += torch.sum(
            torch.stack(
                [
                    torch.distributions.MultivariateNormal(
                        loc=curr_mean_G[:, dd], covariance_matrix=cov_G_list[vv]
                    ).log_prob(curr_G[:, dd])
                    for dd in range(n_spatial_dims)
                ]
            )
        )

    # Log likelihood of Y given G
    LL_Y_expression = torch.sum(
        torch.stack(
            [
                torch.distributions.MultivariateNormal(
                    loc=mean_Y_expression, covariance_matrix=cov_Y_expression
                ).log_prob(Y_expression[:, jj])
                for jj in range(n_genes)
            ]
        )
    )

    LL_Y_histology = torch.sum(
        torch.stack(
            [
                torch.distributions.MultivariateNormal(
                    loc=mean_Y_histology, covariance_matrix=cov_Y_histology
                ).log_prob(Y_histology[:, jj])
                for jj in range(n_channels)
            ]
        )
    )

    # Penalty for preserving pairwise distances between points
    distance_penalty = 0
    for vv in range(n_views):

        # Expression
        curr_distance_mat_X = distance_matrix(
            X_expression[view_idx_expression[vv], :],
            X_expression[view_idx_expression[vv], :],
        )
        curr_distance_mat_G = distance_matrix(
            G_expression[view_idx_expression[vv], :],
            G_expression[view_idx_expression[vv], :],
        )
        curr_penalty = torch.sum(
            torch.square(curr_distance_mat_X - curr_distance_mat_G)
        )
        distance_penalty += curr_penalty

        # Histology
        curr_distance_mat_X = distance_matrix(
            X_histology[view_idx_histology[vv], :],
            X_histology[view_idx_histology[vv], :],
        )
        curr_distance_mat_G = distance_matrix(
            G_histology[view_idx_histology[vv], :],
            G_histology[view_idx_histology[vv], :],
        )
        curr_penalty = torch.sum(
            torch.square(curr_distance_mat_X - curr_distance_mat_G)
        )
        distance_penalty += curr_penalty

    return (
        -LL_G - LL_Y_expression - LL_Y_histology + distance_penalty
    )  # * n_genes / n_channels


class SpatialDataset(Dataset):
    def __init__(self, X_expression, Y_expression, X_histology, Y_histology):
        self.X_expression = X_expression
        self.Y_expression = Y_expression
        self.X_histology = X_histology
        self.Y_histology = Y_histology
        assert X_expression.shape[0] == Y_expression.shape[0]
        assert X_histology.shape[0] == Y_histology.shape[0]

    def __len__(self):
        return self.Y_histology.shape[0]

    def __getitem__(self, idx):
        return self.X_expression, self.Y_expression, self.X_histology, self.Y_histology


if __name__ == "__main__":

    pass
