import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
from models.gpsa import WarpGP
from sklearn.cluster import KMeans
import time

import sys

sys.path.append("..")
from util import rbf_kernel

torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Define model
class VariationalWarpGP(WarpGP):
    def __init__(
        self,
        data_dict,
        m_X_per_view,
        m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_spatial_dims=2,
        n_noise_variance_params=2,
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
        n_latent_gps=1,
        mean_function="identity_fixed",
        mean_penalty_param=0.0,
        fixed_warp_kernel_variances=None,
        fixed_warp_kernel_lengthscales=None,
        fixed_data_kernel_lengthscales=None,
        fixed_view_idx=None,
    ):
        super(VariationalWarpGP, self).__init__(
            data_dict,
            data_init=True,
            n_spatial_dims=2,
            n_noise_variance_params=2,
            kernel_func_warp=kernel_func_warp,
            kernel_func_data=kernel_func_data,
            mean_penalty_param=mean_penalty_param,
            fixed_warp_kernel_variances=fixed_warp_kernel_variances,
            fixed_warp_kernel_lengthscales=fixed_warp_kernel_lengthscales,
            fixed_data_kernel_lengthscales=fixed_data_kernel_lengthscales,
        )

        self.m_X_per_view = m_X_per_view
        self.m_G = m_G
        self.n_latent_gps = n_latent_gps
        self.n_latent_outputs = {}
        for mod in self.modality_names:
            curr_n_latent_outputs = (
                self.n_latent_gps[mod]
                if self.n_latent_gps[mod] is not None
                else self.Ps[mod]
            )
            self.n_latent_outputs[mod] = curr_n_latent_outputs
        self.fixed_view_idx = fixed_view_idx

        if data_init:
            # Initialize inducing locations with a subset of the data
            Xtilde = torch.zeros([self.n_views, self.m_X_per_view, self.n_spatial_dims])
            for ii in range(self.n_views):
                curr_X_spatial_list = []
                for mod in self.modality_names:
                    curr_idx = self.view_idx[mod][ii]
                    curr_modality_and_view_spatial = data_dict[mod]["spatial_coords"][
                        curr_idx, :
                    ]
                    curr_X_spatial_list.append(curr_modality_and_view_spatial)
                curr_X_spatial = torch.cat(curr_X_spatial_list, dim=0)

                kmeans = KMeans(n_clusters=self.m_X_per_view)
                kmeans.fit(curr_X_spatial.detach().numpy())
                Xtilde[ii, :, :] = torch.tensor(kmeans.cluster_centers_)

            self.Xtilde = nn.Parameter(Xtilde.clone())
            # self.Xtilde = Xtilde.clone()

            rand_idx = np.random.choice(
                np.arange(curr_X_spatial.shape[0]),
                size=self.m_G,
                replace=False,
            )

            all_X_spatial = torch.cat(
                [data_dict[mod]["spatial_coords"] for mod in self.modality_names]
            )
            kmeans = KMeans(n_clusters=self.m_G)
            kmeans.fit(all_X_spatial.detach().numpy())
            self.Gtilde = nn.Parameter(torch.tensor(kmeans.cluster_centers_))

        elif grid_init:

            if self.n_spatial_dims == 2:
                xlow, ylow = (
                    data_dict[self.modality_names[0]]["spatial_coords"].numpy().min(0)
                )
                xhigh, yhigh = (
                    data_dict[self.modality_names[0]]["spatial_coords"].numpy().max(0)
                )
                xlimits = [xlow, xhigh]
                ylimits = [ylow, yhigh]
                numticks = np.ceil(np.sqrt(self.m_G)).astype(int)
                self.m_G = numticks ** 2
                self.m_X_per_view = numticks ** 2
                x1s = np.linspace(*xlimits, num=numticks)
                x2s = np.linspace(*ylimits, num=numticks)
                X1, X2 = np.meshgrid(x1s, x2s)
                Xtilde = np.vstack([X1.ravel(), X2.ravel()]).T
                Xtilde_torch = torch.zeros(
                    [self.n_views, Xtilde.shape[0], self.n_spatial_dims]
                )
                for vv in range(self.n_views):
                    Xtilde_torch[vv] = torch.tensor(Xtilde)

                # self.Xtilde = Xtilde_torch.clone()
                # self.Gtilde = torch.tensor(Xtilde).float()
                self.Xtilde = nn.Parameter(Xtilde_torch.clone())
                self.Gtilde = nn.Parameter(torch.tensor(Xtilde).float())

        else:
            # Random initialization of inducing locations
            self.Xtilde = nn.Parameter(
                torch.randn([self.n_views, self.m_X_per_view, self.n_spatial_dims])
            )
            self.Gtilde = nn.Parameter(torch.randn([self.m_G, self.n_spatial_dims]))

        ## Variational covariance parameters
        Omega_sqt_G_list = torch.zeros(
            [self.n_views * self.n_spatial_dims, self.m_X_per_view, self.m_X_per_view]
        )
        for ii in range(self.n_views):
            for jj in range(self.n_spatial_dims):
                Omega_sqt = 0.1 * torch.randn(
                    size=[self.m_X_per_view, self.m_X_per_view]
                )
                # import ipdb; ipdb.set_trace()
                # Omega_sqt_G_list[ii * self.n_views + jj, :, :] = Omega_sqt
                Omega_sqt_G_list[jj * self.n_views + ii, :, :] = Omega_sqt
        self.Omega_sqt_G_list = nn.Parameter(Omega_sqt_G_list)

        Omega_sqt_F_dict = torch.nn.ParameterDict()
        for mod in self.modality_names:
            num_outputs = self.Ps[mod]
            curr_Omega = torch.zeros([self.n_latent_outputs[mod], self.m_G, self.m_G])
            for jj in range(self.n_latent_outputs[mod]):
                Omega_sqt = 0.1 * torch.randn(size=[self.m_G, self.m_G])
                curr_Omega[jj, :, :] = Omega_sqt
            Omega_sqt_F_dict[mod] = nn.Parameter(curr_Omega)
        self.Omega_sqt_F_dict = Omega_sqt_F_dict

        ## Variational mean parameters
        self.delta_G_list = nn.Parameter(self.Xtilde.clone())
        delta_F_dict = torch.nn.ParameterDict()
        for mod in self.modality_names:
            num_outputs = self.Ps[mod]
            curr_delta = nn.Parameter(
                torch.randn(size=[self.m_G, self.n_latent_outputs[mod]])
            )
            delta_F_dict[mod] = curr_delta
        self.delta_F_dict = delta_F_dict

        ## LMC parameters
        self.W_dict = torch.nn.ParameterDict()
        for mod in self.modality_names:
            if self.n_latent_gps[mod] is not None:
                self.W_dict[mod] = nn.Parameter(
                    torch.randn([self.n_latent_gps[mod], self.Ps[mod]])
                )

    def compute_mean_and_var(
        self, Kff_diag, Kuf, Kuu_chol, mu_x, mu_z, delta, Omega_tril
    ):
        alpha_x = torch.cholesky_solve(Kuf, Kuu_chol)

        a_t_Kchol = torch.matmul(alpha_x.transpose(-1, -2), Kuu_chol)
        aKa = torch.sum(torch.square(a_t_Kchol), dim=-1)

        mu_tilde = mu_x.unsqueeze(0) + torch.matmul(
            alpha_x.transpose(-1, -2), delta - mu_z
        )

        if len(alpha_x.shape) == 2:
            a_t_Omega_tril = torch.matmul(
                alpha_x.transpose(-1, -2).unsqueeze(0), Omega_tril
            )
            aOmega_a = torch.sum(torch.square(a_t_Omega_tril), dim=-1)
            Sigma_tilde = Kff_diag - aKa + aOmega_a + self.diagonal_offset
        else:
            a_t_Omega_tril = torch.matmul(
                alpha_x.transpose(-1, -2).unsqueeze(1), Omega_tril.unsqueeze(0)
            )
            aOmega_a = torch.sum(torch.square(a_t_Omega_tril), dim=-1)
            Sigma_tilde = (
                Kff_diag.unsqueeze(1)
                - aKa.unsqueeze(1)
                + aOmega_a
                + self.diagonal_offset
            )

        return mu_tilde, Sigma_tilde + self.diagonal_offset

    def get_Omega_from_Omega_sqt(self, Omega_sqt):
        return (
            torch.matmul(
                Omega_sqt,
                torch.transpose(Omega_sqt, -1, -2),
            )
            + self.diagonal_offset * torch.eye(Omega_sqt.shape[-1])
        )

    def forward(self, X_spatial, view_idx, Ns, S=1, prediction_mode=False):
        self.noise_variance_pos = torch.exp(self.noise_variance) + self.diagonal_offset

        self.mu_z_G = (
            torch.zeros([self.n_views, self.m_X_per_view, self.n_spatial_dims]) * np.nan
        )
        for vv in range(self.n_views):
            self.mu_z_G[vv] = (
                torch.mm(self.Xtilde[vv], self.mean_slopes[vv])
                + self.mean_intercepts[vv]
            )
            if self.fixed_view_idx is not None and (self.fixed_view_idx == vv or vv in self.fixed_view_idx):
                self.mu_z_G[vv] *= 100.0

        self.Kuu_chol_list = (
            torch.zeros([self.n_views, self.m_X_per_view, self.m_X_per_view]) * np.nan
        )
        G_samples = {}
        for mod in self.modality_names:
            G_samples[mod] = torch.zeros([S, Ns[mod], self.n_spatial_dims]) * np.nan

        G_means = {}
        for mod in self.modality_names:
            G_means[mod] = torch.zeros([Ns[mod], self.n_spatial_dims]) * np.nan

        curr_Omega_G = self.get_Omega_from_Omega_sqt(self.Omega_sqt_G_list)

        self.curr_Omega_tril_list = torch.cholesky(curr_Omega_G)

        for vv in range(self.n_views):
            ## If this view is fixed (template-based alignment), then we don't need to sample for it.
            if self.fixed_view_idx is not None and (self.fixed_view_idx == vv or vv in self.fixed_view_idx):

                for mm, mod in enumerate(self.modality_names):
                    observed_X_spatial = X_spatial[mod][view_idx[mod][vv]]
                    G_means[mod][view_idx[mod][vv]] = observed_X_spatial

                    G_samples[mod][:, view_idx[mod][vv], :] = observed_X_spatial

                continue

            kernel_G = lambda x1, x2, diag=False: self.kernel_func_warp(
                x1,
                x2,
                lengthscale_unconstrained=self.warp_kernel_lengthscales[vv],
                output_variance_unconstrained=self.warp_kernel_variances[vv],
                diag=diag,
            )

            ## Collect data from all modalities for this view
            curr_X_spatial_list = []
            curr_n = 0
            curr_mod_idx = []
            for mod in self.modality_names:
                curr_idx = view_idx[mod][vv]
                curr_mod_idx.append(np.arange(curr_n, curr_n + len(curr_idx)))
                curr_n += len(curr_idx)
                curr_modality_and_view_spatial = X_spatial[mod][curr_idx, :]
                curr_X_spatial_list.append(curr_modality_and_view_spatial)

            curr_X_spatial = torch.cat(curr_X_spatial_list, dim=0)

            if len(curr_X_spatial) == 0:
                continue

            curr_X_tilde = self.Xtilde[vv]

            mu_x_G = (
                torch.mm(curr_X_spatial, self.mean_slopes[vv])
                + self.mean_intercepts[vv]
            )

            # Kff_diag = (
            #     kernel_G(curr_X_spatial, curr_X_spatial, diag=True)
            #     + self.diagonal_offset
            # )
            Kff_diag = torch.ones((curr_X_spatial.shape[0])) * torch.exp(
                self.warp_kernel_variances[vv]
            )

            Kuu = kernel_G(
                curr_X_tilde, curr_X_tilde
            ) + self.diagonal_offset * torch.eye(self.m_X_per_view)

            Kuf = kernel_G(curr_X_tilde, curr_X_spatial)

            Kuu_chol = torch.cholesky(Kuu)
            self.Kuu_chol_list[vv, :, :] = Kuu_chol

            mu_tilde, Sigma_tilde = self.compute_mean_and_var(
                Kff_diag,
                Kuf,
                Kuu_chol,
                mu_x_G,
                self.mu_z_G,
                self.delta_G_list,
                self.curr_Omega_tril_list,
            )

            # Sample
            G_marginal_dist = torch.distributions.Normal(
                mu_tilde[vv],
                Sigma_tilde[
                    vv * self.n_spatial_dims : vv * self.n_spatial_dims
                    + self.n_spatial_dims
                ].t(),
            )

            for mm, mod in enumerate(self.modality_names):
                curr_idx = curr_mod_idx[mm]
                G_means[mod][view_idx[mod][vv]] = mu_tilde[vv][curr_idx]

            for ss in range(S):

                curr_G_sample = G_marginal_dist.rsample()
                for mm, mod in enumerate(self.modality_names):
                    curr_idx = curr_mod_idx[mm]
                    G_samples[mod][ss, view_idx[mod][vv]] = curr_G_sample[curr_idx]

        # end = time.time()
        # print("FORWARD 1:", end - start)

        # start = time.time()
        self.curr_Omega_tril_F = {}
        for mod in self.modality_names:
            self.curr_Omega_tril_F[mod] = torch.zeros(
                [self.n_latent_outputs[mod], self.m_G, self.m_G]
            )

        F_samples = {}
        self.F_latent_samples = {}
        self.F_observed_samples = {}
        for mod in self.modality_names:
            F_samples[mod] = torch.zeros([S, Ns[mod], self.n_latent_outputs[mod]])
            self.F_latent_samples[mod] = torch.zeros(
                [S, Ns[mod], self.n_latent_outputs[mod]]
            )
            self.F_observed_samples[mod] = torch.zeros([S, Ns[mod], self.Ps[mod]])

        kernel_F = lambda x1, x2, diag=False: self.kernel_func_data(
            x1,
            x2,
            lengthscale_unconstrained=self.data_kernel_lengthscale,
            output_variance_unconstrained=self.data_kernel_variance,
            diag=diag,
        )

        Kuu = kernel_F(self.Gtilde, self.Gtilde) + self.diagonal_offset * torch.eye(
            self.m_G
        )

        self.Kuu_chol_F = torch.cholesky(Kuu)

        for mod in self.modality_names:

            mu_x_F = torch.zeros([Ns[mod], self.n_latent_outputs[mod]])
            mu_z_F = torch.zeros([self.m_G, self.n_latent_outputs[mod]])

            # Kff_diag = (
            #     kernel_F(G_samples[mod], G_samples[mod], diag=True)
            #     + self.diagonal_offset
            # )
            Kff_diag = torch.ones((G_samples[mod].shape[:2])) * torch.exp(
                self.data_kernel_variance
            )

            Kuf = kernel_F(self.Gtilde, G_samples[mod])
            # start = time.time()
            curr_Omega = self.get_Omega_from_Omega_sqt(self.Omega_sqt_F_dict[mod])

            self.curr_Omega_tril_F[mod] = torch.cholesky(curr_Omega)
            mu_tilde, Sigma_tilde = self.compute_mean_and_var(
                Kff_diag,
                Kuf,
                self.Kuu_chol_F,
                mu_x_F,
                mu_z_F,
                self.delta_F_dict[mod],
                self.curr_Omega_tril_F[mod],
            )

            eps = torch.randn(mu_tilde.shape)
            curr_F_latent_samples = (
                mu_tilde + torch.sqrt(torch.transpose(Sigma_tilde, 1, 2)) * eps
            )

            if self.n_latent_gps[mod] is not None:
                curr_W = self.W_dict[mod]
                F_observed_mean = torch.matmul(curr_F_latent_samples, curr_W)
            else:
                F_observed_mean = curr_F_latent_samples

            self.F_latent_samples[mod] = curr_F_latent_samples
            self.F_observed_samples[mod] = F_observed_mean

        return G_means, G_samples, self.F_latent_samples, self.F_observed_samples

    def loss_fn(self, data_dict, F_samples):
        # This is the negative (approximate) ELBO

        # KL terms
        KL_div = 0

        ## G
        for vv in range(self.n_views):
            if self.fixed_view_idx is not None and (self.fixed_view_idx == vv or vv in self.fixed_view_idx):
                continue
            for jj in range(self.n_spatial_dims):
                qu = torch.distributions.MultivariateNormal(
                    loc=self.delta_G_list[vv, :, jj],
                    # scale_tril=self.curr_Omega_tril_list[vv * self.n_views + jj, :, :],
                    scale_tril=self.curr_Omega_tril_list[jj * self.n_views + vv, :, :],
                )
                pu = torch.distributions.MultivariateNormal(
                    loc=self.mu_z_G[vv, :, jj],
                    scale_tril=self.Kuu_chol_list[vv, :, :],
                )
                curr_KL_div = torch.distributions.kl.kl_divergence(qu, pu)

                KL_div += curr_KL_div

        ## F
        LL = 0
        pu = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.m_G), scale_tril=self.Kuu_chol_F
        )
        for mm, mod in enumerate(self.modality_names):
            qu = torch.distributions.MultivariateNormal(
                loc=self.delta_F_dict[mod].t(),
                scale_tril=self.curr_Omega_tril_F[mod],
            )

            curr_KL_div = torch.distributions.kl.kl_divergence(qu, pu)
            KL_div += curr_KL_div.sum()

            Y_distribution = torch.distributions.Normal(
                loc=F_samples[mod],
                scale=self.noise_variance_pos[-self.n_modalities + mm],
            )
            S = F_samples[mod].shape[0]

            LL += Y_distribution.log_prob(data_dict[mod]["outputs"]).sum() / S

        return -LL + KL_div


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
