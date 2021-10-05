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
        mean_function="identity_fixed",
        mean_penalty_param=0.,
        fixed_warp_kernel_variances=None,
        fixed_warp_kernel_lengthscales=None,
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
        # self.Ns = {}
        # self.Ps = {}
        # self.n_samples_lists = {}
        # self.view_idx = {}
        # self.n_total = 0
        
        view_idx, Ns, Ps, n_total = self.create_view_idx_dict(data_dict)
        self.view_idx = view_idx
        self.Ns = Ns
        self.Ps = Ps
        self.n_total = n_total
        # import ipdb; ipdb.set_trace()

        ## Number of kernel parameters:
        ##		- 2 parameters for each view for warp GP (lengthscale and variance)
        ##		- 2 parameters for observation GP (lengthscale and variance)
        self.n_kernel_params = 2 * self.n_views + 2
        self.n_noise_variance_params = n_noise_variance_params
        self.kernel_func = kernel_func

        ## Parameters

        # self.G_param = nn.Parameter(G_init[self.view_idx[0], :])
        self.noise_variance = nn.Parameter(torch.randn([self.n_noise_variance_params]))

        if fixed_warp_kernel_variances is None:
            self.warp_kernel_variances = nn.Parameter(torch.randn([self.n_kernel_params // 2 - 1]))
        else:
            self.warp_kernel_variances = torch.log(torch.tensor(fixed_warp_kernel_variances))

        if fixed_warp_kernel_lengthscales is None:
            self.warp_kernel_lengthscales = nn.Parameter(
                torch.randn([self.n_kernel_params // 2 - 1])
            )
        else:
            self.warp_kernel_lengthscales = torch.log(torch.tensor(fixed_warp_kernel_lengthscales))

        self.data_kernel_variance = nn.Parameter(torch.randn(1))
        self.data_kernel_lengthscale = nn.Parameter(torch.randn(1))
        # self.data_kernel_variance = torch.tensor(1)
        # self.data_kernel_lengthscale = torch.tensor(1)
    
        
        if mean_function == "identity_fixed":
            self.mean_slopes = torch.eye(self.n_spatial_dims).unsqueeze(0).repeat(self.n_views, 1, 1)
            self.mean_intercepts = torch.zeros([self.n_views, self.n_spatial_dims])
        elif mean_function == "identity_initialized":
            self.mean_slopes = nn.Parameter(
                torch.randn([self.n_views, self.n_spatial_dims, self.n_spatial_dims])
            )
            self.mean_intercepts = nn.Parameter(
                torch.zeros([self.n_views, self.n_spatial_dims])
            )
        else:
            self.mean_slopes = nn.Parameter(
                torch.eye(self.n_spatial_dims).unsqueeze(0).repeat(self.n_views, 1, 1)
            )
            self.mean_intercepts = nn.Parameter(
                torch.randn([self.n_views, self.n_spatial_dims]) * 0.1
            )
        
        # self.mean_intercepts = torch.zeros([self.n_views, self.n_spatial_dims])
        self.diagonal_offset = 1e-5

    def create_view_idx_dict(self, data_dict):

        view_idx, Ns, Ps = {}, {}, {}
        n_total = 0
        for mod in self.modality_names:
            n_samples_list = data_dict[mod]["n_samples_list"]
            # self.n_samples_lists[mod] = n_samples_list
            curr_N = np.sum(n_samples_list)
            Ns[mod] = curr_N
            n_total += curr_N
            Ps[mod] = data_dict[mod]["outputs"].shape[1]

            # Compute the indices of each view for each modality
            cumulative_sums = np.cumsum(n_samples_list)
            cumulative_sums = np.insert(cumulative_sums, 0, 0)
            curr_view_idx = [
                    np.arange(cumulative_sums[ii], cumulative_sums[ii + 1])
                    for ii in range(self.n_views)
                ]
            # import ipdb; ipdb.set_trace()
            view_idx[mod] = curr_view_idx
            # import ipdb; ipdb.set_trace()

        return view_idx, Ns, Ps, n_total

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
