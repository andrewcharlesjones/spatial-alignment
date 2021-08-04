import torch
import pyro.contrib.gp as gp
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy
from warp_gp import WarpGP


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))

# Define model
class TwoLayerWarpGP(WarpGP):
	def __init__(
		self,
		data_dict,
		data_init=True,
		n_spatial_dims=2,
		n_noise_variance_params=1,
		kernel_func=gp.kernels.RBF,
		distance_penalty=True,
	):
		super(TwoLayerWarpGP, self).__init__(data_dict,
											 data_init=True,
											 n_spatial_dims=2,
											 n_noise_variance_params=1,
											 kernel_func=gp.kernels.RBF)
		self.modality_names = list(data_dict.keys())
		self.n_modalities = len(self.modality_names)
		self.distance_penalty = distance_penalty

		self.Gs = torch.nn.ParameterDict()
		for mod in self.modality_names:
			if data_init:
				curr_G = nn.Parameter(data_dict[mod]["spatial_coords"].clone())
			else:
				curr_G = nn.Parameter(torch.randn([self.Ns[mod], self.n_spatial_dims]))
			self.Gs[mod] = curr_G

	def forward(self, X_spatial):
		noise_variance_pos = torch.exp(self.noise_variance) + 1e-4
		kernel_variances_pos = torch.exp(self.kernel_variances)
		kernel_lengthscales_pos = torch.exp(self.kernel_lengthscales)

		###############################################
		##### Compute means and covariances for G #####
		###############################################
		means_G_list = []
		covs_G_list = []
		for vv in range(self.n_views):
			curr_X_spatial_list = []
			curr_n = 0
			for mod in self.modality_names:
				curr_idx = self.view_idx[mod][vv]
				curr_n += len(curr_idx)
				curr_modality_and_view_spatial = X_spatial[mod][curr_idx, :]
				curr_X_spatial_list.append(curr_modality_and_view_spatial)
			curr_X_spatial = torch.cat(curr_X_spatial_list, dim=0)

			## Make mean
			mean_G = (
				torch.matmul(curr_X_spatial, self.mean_slopes[vv])
				+ self.mean_intercepts[vv]
			)
			means_G_list.append(mean_G)

			## Make covariance
			kernel_G = self.kernel_func(
				input_dim=self.n_spatial_dims,
				variance=kernel_variances_pos[vv],
				lengthscale=kernel_lengthscales_pos[vv],
			)
			cov_G = kernel_G(
				curr_X_spatial, curr_X_spatial
			) + self.diagonal_offset * torch.eye(curr_n)
			covs_G_list.append(cov_G)

		###############################################
		##### Compute means and covariances for Y #####
		###############################################
		kernel_Y = self.kernel_func(
			input_dim=self.n_spatial_dims,
			variance=kernel_variances_pos[-1],
			lengthscale=kernel_lengthscales_pos[-1],
		)
		means_Y = {}
		covs_Y = {}
		for mod in self.modality_names:

			## Zero mean
			mean_Y = torch.zeros(self.Ns[mod])
			means_Y[mod] = mean_Y

			## Covariance based on G
			cov_Y = kernel_Y(
				self.Gs[mod], self.Gs[mod]
			) + noise_variance_pos * torch.eye(self.Ns[mod])
			covs_Y[mod] = cov_Y

		return self.Gs, means_G_list, covs_G_list, means_Y, covs_Y

	def loss_fn(self, data_dict, Gs, means_G_list, covs_G_list, means_Y, covs_Y):
		# Log likelihood of G given X (warp likelihood)
		LL_G = 0
		for vv in range(self.n_views):
			curr_view_idxs = np.array(
				[self.view_idx[mod][vv] for mod in self.modality_names]
			)
			curr_G = torch.cat(
				[
					Gs[mod][curr_view_idxs[ii]]
					for ii, mod in enumerate(self.modality_names)
				]
			)

			curr_mean_G = means_G_list[vv]
			curr_cov_G = covs_G_list[vv]

			LL_G += torch.sum(
				torch.stack(
					[
						torch.distributions.MultivariateNormal(
							loc=curr_mean_G[:, dd], covariance_matrix=curr_cov_G
						).log_prob(curr_G[:, dd])
						for dd in range(self.n_spatial_dims)
					]
				)
			)

		# Log likelihood of Y given G
		LL_Y = 0
		for mod in self.modality_names:
			curr_Y = data_dict[mod]["outputs"]
			curr_mean_Y = means_Y[mod]
			curr_cov_Y = covs_Y[mod]
			LL_Y += torch.sum(
				torch.stack(
					[
						torch.distributions.MultivariateNormal(
							loc=curr_mean_Y, covariance_matrix=curr_cov_Y
						).log_prob(curr_Y[:, jj])
						for jj in range(self.Ps[mod])
					]
				)
			)

		# Penalty for preserving pairwise distances between points
		distance_penalty = 0
		if self.distance_penalty:
			for vv in range(self.n_views):
				for mod in self.modality_names:

					## Compute distance matrices
					curr_view_idxs = np.array(
						[self.view_idx[mod][vv] for mod in self.modality_names]
					)
					curr_X = torch.cat(
						[
							data_dict[mod]["spatial_coords"][curr_view_idxs[ii]]
							for ii, mod in enumerate(self.modality_names)
						]
					)
					curr_G = torch.cat(
						[
							Gs[mod][curr_view_idxs[ii]]
							for ii, mod in enumerate(self.modality_names)
						]
					)
					curr_distance_mat_X = distance_matrix(curr_X, curr_X)
					curr_distance_mat_G = distance_matrix(curr_G, curr_G)

					## Penalize deviations from original pairwise distances
					curr_penalty = torch.sum(
						torch.square(curr_distance_mat_X - curr_distance_mat_G)
					)
					distance_penalty += curr_penalty

		return -LL_G - LL_Y + distance_penalty


def distance_matrix(X, Y):
	squared_diffs = torch.square(torch.unsqueeze(X, 0) - torch.unsqueeze(Y, 1))
	squared_distances = torch.sum(squared_diffs, dim=2)
	return squared_distances


if __name__ == "__main__":

	pass
