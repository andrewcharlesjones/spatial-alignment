import torch
import pyro.contrib.gp as gp
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))

# Define model
class WarpGP(nn.Module):
    def __init__(
        self,
        data_dict,
        data_init=True,
        n_spatial_dims=2,
        n_noise_variance_params=2,
        kernel_func=gp.kernels.RBF,
        mean_penalty_param=0.,
    ):
        super(WarpGP, self).__init__()
        self.modality_names = list(data_dict.keys())
        self.n_modalities = len(self.modality_names)
        self.mean_penalty_param = mean_penalty_param

        ## Make sure all modalities have the same number of "views"
        n_views = np.unique(
            np.array(
                [len(data_dict[mod]["n_samples_list"]) for mod in self.modality_names]
            )
        )
        if len(n_views) != 1:
            raise ValueError("Each modality must have the same number of views.")
        self.n_views = n_views[0]

        ## Make sure all modalities have the same domain for the spatial coordinates
        n_spatial_dims = np.unique(
            np.array(
                [
                    data_dict[mod]["spatial_coords"].shape[1]
                    for mod in self.modality_names
                ]
            )
        )
        if len(n_spatial_dims) != 1:
            raise ValueError(
                "Each modality must have the same number of spatial dimensions."
            )
        self.n_spatial_dims = n_spatial_dims[0]
        self.Ns = {}
        self.Ps = {}
        self.n_samples_lists = {}
        self.view_idx = {}
        self.n_total = 0
        for mod in self.modality_names:
            n_samples_list = data_dict[mod]["n_samples_list"]
            self.n_samples_lists[mod] = n_samples_list
            curr_N = np.sum(n_samples_list)
            self.Ns[mod] = curr_N
            self.n_total += curr_N
            self.Ps[mod] = data_dict[mod]["outputs"].shape[1]

            # Compute the indices of each view for each modality
            cumulative_sums = np.cumsum(n_samples_list)
            cumulative_sums = np.insert(cumulative_sums, 0, 0)
            view_idx = np.array(
                [
                    np.arange(cumulative_sums[ii], cumulative_sums[ii + 1])
                    for ii in range(self.n_views)
                ]
            )
            self.view_idx[mod] = view_idx

        ## Number of kernel parameters:
        ##		- 2 parameters for each view for warp GP (lengthscale and variance)
        ##		- 2 parameters for observation GP (lengthscale and variance)
        self.n_kernel_params = 2 * self.n_views + 2
        self.n_noise_variance_params = n_noise_variance_params
        self.kernel_func = kernel_func

        ## Parameters

        # self.G_param = nn.Parameter(G_init[self.view_idx[0], :])
        self.noise_variance = nn.Parameter(torch.randn([self.n_noise_variance_params]))
        self.kernel_variances = nn.Parameter(torch.randn([self.n_kernel_params // 2]) - 5)
        self.kernel_lengthscales = nn.Parameter(
            torch.randn([self.n_kernel_params // 2])
        )

        self.mean_slopes = nn.Parameter(
            torch.eye(self.n_spatial_dims).unsqueeze(0).repeat(self.n_views, 1, 1)
        )
        # self.mean_slopes = nn.Parameter(
        #     torch.randn([self.n_views, self.n_spatial_dims, self.n_spatial_dims])
        # )
        self.mean_intercepts = nn.Parameter(
            torch.randn([self.n_views, self.n_spatial_dims]) * 0.1
        )
        # self.mean_intercepts = torch.zeros([self.n_views, self.n_spatial_dims])
        self.diagonal_offset = 1e-5

    def compute_mean_penalty(self):
        return self.mean_penalty_param * torch.mean(
            torch.square(
                self.mean_slopes
                - torch.eye(self.n_spatial_dims).unsqueeze(0).repeat(self.n_views, 1, 1)
            )
        )

    def forward(self, X_spatial):
        raise (NotImplementedError)

    def loss_fn(self, data_dict, Gs, means_G_list, covs_G_list, means_Y, covs_Y):
        raise (NotImplementedError)


def distance_matrix(X, Y):
    squared_diffs = torch.square(torch.unsqueeze(X, 0) - torch.unsqueeze(Y, 1))
    squared_distances = torch.sum(squared_diffs, dim=2)
    return squared_distances


if __name__ == "__main__":

    pass
