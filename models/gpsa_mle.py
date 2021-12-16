import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
from models.gpsa import WarpGP

import sys

sys.path.append("..")
from util import rbf_kernel

torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Define model
class WarpGPMLE(WarpGP):
    def __init__(
        self,
        data_dict,
        n_spatial_dims=2,
        n_noise_variance_params=1,
        kernel_func=rbf_kernel,
        n_latent_gps=1,
        mean_function="identity_fixed",
        mean_penalty_param=0.0,
        fixed_warp_kernel_variances=None,
        fixed_warp_kernel_lengthscales=None,
        fixed_data_kernel_lengthscales=None,
        fixed_view_idx=None,
    ):
        super(WarpGPMLE, self).__init__(
            data_dict,
            data_init=True,
            n_spatial_dims=2,
            n_noise_variance_params=2,
            kernel_func=rbf_kernel,
            mean_penalty_param=mean_penalty_param,
            fixed_warp_kernel_variances=fixed_warp_kernel_variances,
            fixed_warp_kernel_lengthscales=fixed_warp_kernel_lengthscales,
            fixed_data_kernel_lengthscales=fixed_data_kernel_lengthscales,
        )

        self.n_latent_gps = n_latent_gps
        self.n_latent_outputs = {}
        for mod in self.modality_names:
            curr_n_latent_outputs = (
                self.n_latent_gps if self.n_latent_gps is not None else self.Ps[mod]
            )
            self.n_latent_outputs[mod] = curr_n_latent_outputs
        self.fixed_view_idx = fixed_view_idx

        self.G = torch.nn.ParameterDict()
        for mod in self.modality_names:
            self.G[mod] = torch.nn.Parameter(data_dict[mod]["spatial_coords"])

        if self.n_latent_gps is not None:
            self.W_dict = torch.nn.ParameterDict()
            for mod in self.modality_names:
                self.W_dict[mod] = nn.Parameter(
                    torch.randn([self.n_latent_gps, self.Ps[mod]])
                )

    def forward(self, X_spatial, view_idx, Ns, prediction_mode=False):
        self.noise_variance_pos = torch.exp(self.noise_variance) + 1e-4

        return

    def loss_fn(self, X_spatial, view_idx, data_dict):
        ## Compute likelihood of warp
        G_LL = 0
        for vv in range(self.n_views):
            if self.fixed_view_idx == vv:
                continue

            curr_X_spatial_list = []
            curr_G_list = []
            curr_n = 0
            curr_mod_idx = []
            for mod in self.modality_names:
                curr_idx = view_idx[mod][vv]
                curr_mod_idx.append(np.arange(curr_n, curr_n + len(curr_idx)))
                curr_n += len(curr_idx)
                curr_modality_and_view_spatial = X_spatial[mod][curr_idx, :]
                curr_X_spatial_list.append(curr_modality_and_view_spatial)

                curr_G_list.append(self.G[mod][curr_idx, :])

            curr_X_spatial = torch.cat(curr_X_spatial_list, dim=0)
            curr_G_list = torch.cat(curr_G_list, dim=0)

            kernel_G = lambda x1, x2: rbf_kernel(
                x1,
                x2,
                lengthscale_unconstrained=self.warp_kernel_lengthscales[vv],
                output_variance_unconstrained=self.warp_kernel_variances[vv],
            )

            mean = (
                torch.matmul(curr_X_spatial, self.mean_slopes[vv])
                + self.mean_intercepts[vv]
            )
            K_XX = kernel_G(curr_X_spatial, curr_X_spatial)

            for dd in range(self.n_spatial_dims):
                G_distribution = torch.distributions.MultivariateNormal(
                    loc=mean[:, dd],
                    covariance_matrix=K_XX + 1e-5 * torch.eye(K_XX.shape[0]),
                )
                G_LL += G_distribution.log_prob(curr_G_list[:, dd]).sum()

        ## Compute likelihood of data
        kernel_F = lambda x1, x2: rbf_kernel(
            x1,
            x2,
            lengthscale_unconstrained=self.data_kernel_lengthscale,
            output_variance_unconstrained=self.data_kernel_variance,
        )

        Y_LL = 0
        for mod in self.modality_names:
            # import ipdb; ipdb.set_trace()
            if self.fixed_view_idx is not None:
                curr_G = torch.zeros(self.G[mod].shape)
                fixed_G = self.G[mod].detach()[view_idx[mod][vv]]
                fixed_G.requires_grad = False

                for vv in range(self.n_views):
                    if vv == self.fixed_view_idx:
                        curr_G[view_idx[mod][vv]] += fixed_G  # [view_idx[mod][vv]]
                    else:
                        curr_G[view_idx[mod][vv]] += self.G[mod][view_idx[mod][vv]]
                # import ipdb; ipdb.set_trace()
            else:
                curr_G = self.G[mod]

            noise = self.noise_variance_pos[1] * torch.eye(curr_G.shape[0])
            K_GG = kernel_F(curr_G, curr_G) + noise
            Y_distribution = torch.distributions.MultivariateNormal(
                loc=torch.zeros(K_GG.shape[0]), covariance_matrix=K_GG
            )
            for jj in range(self.Ps[mod]):
                Y_LL += Y_distribution.log_prob(data_dict[mod]["outputs"][:, jj]).sum()

        LL = G_LL + Y_LL
        # print(G_LL, Y_LL)
        return -LL


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
    pass
