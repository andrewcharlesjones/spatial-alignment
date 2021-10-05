import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.contrib.gp as gp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# from util import make_pinwheel
import seaborn as sns
from warp_gp import WarpGP

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Define model
class VariationalLMCWarpGP(WarpGP):
    # def __init__(self, X, view_idx, n, n_spatial_dims, m_X_per_view, m_G):
    def __init__(
        self,
        data_dict,
        m_X_per_view,
        m_G,
        data_init=True,
        minmax_init=False,
        n_spatial_dims=2,
        n_noise_variance_params=1,
        kernel_func=gp.kernels.RBF,
        n_latent_gps=1,
    ):
        super(VariationalLMCWarpGP, self).__init__(
            data_dict,
            data_init=True,
            n_spatial_dims=2,
            n_noise_variance_params=1,
            kernel_func=gp.kernels.RBF,
        )

        self.m_X_per_view = m_X_per_view
        self.m_G = m_G
        self.n_latent_gps = n_latent_gps

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
                rand_idx = np.random.choice(
                    np.arange(curr_X_spatial.shape[0]),
                    size=self.m_X_per_view,
                    replace=False,
                )
                Xtilde[ii, :, :] = curr_X_spatial[rand_idx]
            # self.Xtilde = nn.Parameter(Xtilde)
            self.Xtilde = Xtilde.clone()
        elif minmax_init:

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
                curr_inducing_locs = torch.linspace(
                    torch.min(curr_X_spatial),
                    torch.max(curr_X_spatial),
                    self.m_X_per_view,
                )
                Xtilde[ii, :, 0] = curr_inducing_locs
                self.Xtilde = Xtilde.clone()
        else:
            # Random initialization of inducing locations
            self.Xtilde = nn.Parameter(
                torch.randn([self.n_views, self.m_X_per_view, self.n_spatial_dims])
            )
        self.Gtilde = nn.Parameter(torch.randn([self.m_G, self.n_spatial_dims]))
        Omega_sqt_G_list = torch.zeros(
            [self.n_views, self.n_spatial_dims, self.m_X_per_view, self.m_X_per_view]
        )
        for ii in range(self.n_views):
            for jj in range(self.n_spatial_dims):
                Omega_sqt = 0.01 * torch.randn(size=[m_X_per_view, m_X_per_view])
                Omega_sqt_G_list[ii, jj, :, :] = Omega_sqt
        self.Omega_sqt_G_list = nn.Parameter(Omega_sqt_G_list)

        Omega_sqt_F_dict = torch.nn.ParameterDict()
        for mod in self.modality_names:
            num_outputs = self.n_latent_gps
            curr_Omega = torch.zeros([num_outputs, m_G, m_G])
            for jj in range(num_outputs):
                Omega_sqt = 0.01 * torch.randn(size=[m_G, m_G])
                curr_Omega[jj, :, :] = Omega_sqt
            Omega_sqt_F_dict[mod] = nn.Parameter(curr_Omega)
        self.Omega_sqt_F_dict = Omega_sqt_F_dict

        # import ipdb; ipdb.set_trace()
        # self.delta_list = nn.Parameter(
        #     torch.randn([self.n_views, self.m_X_per_view, self.n_spatial_dims])
        # )
        self.delta_list = nn.Parameter(self.Xtilde.clone())

        delta_F_dict = torch.nn.ParameterDict()
        for mod in self.modality_names:
            num_outputs = self.n_latent_gps
            curr_delta = nn.Parameter(0.1 * torch.randn(size=[m_G, num_outputs]))
            delta_F_dict[mod] = curr_delta
        self.delta_F_dict = delta_F_dict

        self.W_dict = torch.nn.ParameterDict()
        for mod in self.modality_names:
            self.W_dict[mod] = nn.Parameter(
                torch.randn([self.n_latent_gps, self.Ps[mod]])
            )

    def forward(self, X_spatial, S=1):
        self.noise_variance_pos = torch.exp(self.noise_variance) + 1e-4
        kernel_variances_pos = torch.exp(self.kernel_variances)
        kernel_lengthscales_pos = torch.exp(self.kernel_lengthscales)
        self.kernel_G = gp.kernels.RBF(
            input_dim=1,
            variance=kernel_variances_pos[0],
            lengthscale=kernel_lengthscales_pos[0],
        )
        # n_total = X_spatial.shape[0]

        self.mu_z_G = torch.zeros(
            [self.n_views, self.m_X_per_view, self.n_spatial_dims]
        )
        for vv in range(self.n_views):
            self.mu_z_G[vv] = (
                torch.mm(self.Xtilde[vv], self.mean_slopes[vv])
                + self.mean_intercepts[vv]
            )

        self.curr_Omega_tril_list = torch.zeros(
            [self.n_views, self.n_spatial_dims, self.m_X_per_view, self.m_X_per_view]
        )
        self.Kuu_chol_list = torch.zeros(
            [self.n_views, self.n_spatial_dims, self.m_X_per_view, self.m_X_per_view]
        )
        G_samples = {}
        for mod in self.modality_names:
            G_samples[mod] = torch.zeros([S, self.Ns[mod], self.n_spatial_dims])

        self.G_means = {}
        for mod in self.modality_names:
            self.G_means[mod] = torch.zeros([self.Ns[mod], self.n_spatial_dims])
        for ii in range(self.n_views):

            ## Collect data from all modalities for this view
            curr_X_spatial_list = []
            curr_n = 0
            curr_mod_idx = []
            for mod in self.modality_names:
                curr_idx = self.view_idx[mod][ii]
                curr_mod_idx.append(np.arange(curr_n, len(curr_idx)))
                curr_n += len(curr_idx)
                curr_modality_and_view_spatial = X_spatial[mod][curr_idx, :]
                curr_X_spatial_list.append(curr_modality_and_view_spatial)
            curr_X_spatial = torch.cat(curr_X_spatial_list, dim=0)
            curr_X_tilde = self.Xtilde[ii]

            mu_x_G = (
                torch.mm(curr_X_spatial, self.mean_slopes[ii])
                + self.mean_intercepts[ii]
            )

            for jj in range(self.n_spatial_dims):

                curr_delta = self.delta_list[ii, :, jj]
                curr_mu_z_G = self.mu_z_G[ii, :, jj]
                curr_mu_x_G = mu_x_G[:, jj]

                curr_Omega_sqt_G = self.Omega_sqt_G_list[ii, jj, :, :].clone()
                curr_Omega_G = torch.matmul(
                    curr_Omega_sqt_G, curr_Omega_sqt_G.t()
                ) + 1e-3 * torch.eye(self.m_X_per_view)
                curr_Omega_tril = torch.cholesky(curr_Omega_G)
                self.curr_Omega_tril_list[ii, jj, :, :] = curr_Omega_tril

                ## Sample G from p(G | X, Xtilde)
                ## 		- Note that this is the distribution with
                ## 		  the pseudo-outputs marginalized out
                Kff_diag = (
                    self.kernel_G(curr_X_spatial, curr_X_spatial, diag=True)
                    + self.diagonal_offset
                )
                Kuu = self.kernel_G(
                    curr_X_tilde, curr_X_tilde
                ) + self.diagonal_offset * torch.eye(self.m_X_per_view)
                Kuu_chol = torch.cholesky(Kuu)
                self.Kuu_chol_list[ii, jj, :, :] = Kuu_chol
                Kuf = self.kernel_G(curr_X_tilde, curr_X_spatial)

                # TODO: make these true mean functions (linear or something)
                # mu_x = torch.zeros(curr_X_spatial.shape[0])
                # self.mu_z_G = torch.zeros(m)

                alpha_x = torch.cholesky_solve(Kuf, Kuu_chol)
                mu_tilde = curr_mu_x_G + torch.matmul(
                    alpha_x.t(), curr_delta - curr_mu_z_G
                )

                a_t_Kchol = torch.matmul(alpha_x.t(), Kuu_chol)
                aKa = torch.sum(torch.square(a_t_Kchol), dim=1)
                a_t_Omega_tril = torch.matmul(alpha_x.t(), curr_Omega_tril)
                aOmega_a = torch.sum(torch.square(a_t_Omega_tril), dim=1)
                Sigma_tilde = Kff_diag - aKa + aOmega_a + self.diagonal_offset

                # Sample
                G_marginal_dist = torch.distributions.MultivariateNormal(
                    loc=mu_tilde, covariance_matrix=torch.diag(Sigma_tilde)
                )
                curr_G_samples = G_marginal_dist.rsample(sample_shape=[S])

                for mm, mod in enumerate(self.modality_names):

                    curr_idx = curr_mod_idx[mm]
                    G_samples[mod][:, self.view_idx[mod][ii], jj] = curr_G_samples[
                        :, curr_idx
                    ]

                    self.G_means[mod][self.view_idx[mod][ii], jj] = mu_tilde

        ## Sample F from p(F | G, Gtilde)
        # self.mu_z_F = torch.zeros([self.Gtilde.shape]) # torch.matmul(self.Gtilde, self.mean_slopes) + self.mean_intercepts

        self.kernel_F = gp.kernels.RBF(
            input_dim=self.n_spatial_dims,
            variance=kernel_variances_pos[-1],
            lengthscale=kernel_lengthscales_pos[-1],
        )

        self.curr_Omega_tril_F = {}
        for mod in self.modality_names:
            self.curr_Omega_tril_F[mod] = torch.zeros(
                [self.n_latent_gps, self.m_G, self.m_G]
            )
        latent_F_samples = {}
        for mod in self.modality_names:
            latent_F_samples[mod] = torch.zeros([S, self.Ns[mod], self.n_latent_gps])
        observed_F_samples = {}
        for mod in self.modality_names:
            observed_F_samples[mod] = torch.zeros([S, self.Ns[mod], self.Ps[mod]])

        for ss in range(S):
            for mod in self.modality_names:
                curr_G = G_samples[mod][ss, :, :].clone()

                self.mu_x_F = torch.zeros([self.Ns[mod], self.n_latent_gps])
                self.mu_z_F = torch.zeros([self.m_G, self.n_latent_gps])

                Kff_diag = self.kernel_F(curr_G, diag=True) + self.diagonal_offset

                Kuu = self.kernel_F(
                    self.Gtilde, self.Gtilde
                ) + self.diagonal_offset * torch.eye(self.m_G)
                self.Kuu_chol = torch.cholesky(Kuu)
                Kuf = self.kernel_F(self.Gtilde, curr_G)

                for jj in range(self.n_latent_gps):
                    curr_Omega_sqt = self.Omega_sqt_F_dict[mod][jj, :, :].clone()
                    curr_Omega = torch.matmul(
                        curr_Omega_sqt, curr_Omega_sqt.t()
                    ) + self.diagonal_offset * torch.eye(self.m_G)
                    self.curr_Omega_tril_F[mod][jj, :, :] = torch.cholesky(curr_Omega)

                    ## Sample F from p(G | X, Xtilde)
                    ## 		- Note that this is the distribution with
                    ## 		  the pseudo-outputs marginalized out

                    # TODO: make these true mean functions (linear or something)
                    # mu_x = torch.zeros(curr_G.shape[0])
                    # self.mu_z = torch.zeros(m*n_views)

                    alpha_x = torch.cholesky_solve(Kuf, self.Kuu_chol)
                    mu_tilde = self.mu_x_F[:, jj] + torch.matmul(
                        alpha_x.t(), self.delta_F_dict[mod][:, jj] - self.mu_z_F[:, jj]
                    )
                    a_t_Kchol = torch.matmul(alpha_x.t(), self.Kuu_chol)
                    aKa = torch.sum(torch.square(a_t_Kchol), dim=1)
                    a_t_Omega_tril = torch.matmul(
                        alpha_x.t(), self.curr_Omega_tril_F[mod][jj, :, :].clone()
                    )
                    aOmega_a = torch.sum(torch.square(a_t_Omega_tril), dim=1)
                    Sigma_tilde = Kff_diag - aKa + aOmega_a + self.diagonal_offset
                    self.Sigma_tilde = Sigma_tilde

                    # Sample
                    F_marginal_dist = torch.distributions.MultivariateNormal(
                        loc=mu_tilde, covariance_matrix=torch.diag(Sigma_tilde)
                    )
                    curr_latent_F_samples = F_marginal_dist.rsample(sample_shape=[1])
                    latent_F_samples[mod][:, :, jj] = curr_latent_F_samples

                    observed_F_mean = torch.matmul(
                        curr_latent_F_samples.t(), self.W_dict[mod]
                    )
                    F_observed_dist = torch.distributions.Normal(
                        observed_F_mean, self.noise_variance_pos
                    )
                    observed_F_samples[mod][:, :, :] = F_observed_dist.rsample([1])
                    # import ipdb; ipdb.set_trace()

        return G_samples, latent_F_samples, observed_F_samples, mu_tilde

    def loss_fn(self, data_dict, observed_F_samples):
        # This is the negative (approximate) ELBO

        # KL terms
        KL_div = 0

        ## G
        for ii in range(self.n_views):
            for jj in range(self.n_spatial_dims):
                qu = torch.distributions.MultivariateNormal(
                    loc=self.delta_list[ii, :, jj],
                    scale_tril=self.curr_Omega_tril_list[ii, jj, :, :],
                )
                pu = torch.distributions.MultivariateNormal(
                    loc=self.mu_z_G[ii, :, jj],
                    scale_tril=self.Kuu_chol_list[ii, jj, :, :],
                )
                KL_div += torch.distributions.kl.kl_divergence(qu, pu)  # / self.n_views

        ## F
        LL = 0
        for mod in self.modality_names:
            for jj in range(self.n_latent_gps):
                qu = torch.distributions.MultivariateNormal(
                    loc=self.delta_F_dict[mod][:, jj],
                    scale_tril=self.curr_Omega_tril_F[mod][jj, :, :],
                )
                pu = torch.distributions.MultivariateNormal(
                    loc=self.mu_z_F[:, jj], scale_tril=self.Kuu_chol
                )
                KL_div += (
                    torch.distributions.kl.kl_divergence(qu, pu) / self.n_latent_gps
                )

            for jj in range(self.Ps[mod]):
                # Log likelihood
                # import ipdb; ipdb.set_trace()
                Y_distribution = torch.distributions.Normal(
                    loc=observed_F_samples[mod][:, :, jj], scale=self.noise_variance_pos
                )
                LL += Y_distribution.log_prob(data_dict[mod]["outputs"][:, jj])

        KL_loss = self.n_spatial_dims * KL_div
        LL_loss = -torch.sum(torch.mean(LL, dim=0))

        mean_penalty = self.compute_mean_penalty()
        return LL_loss + KL_loss + mean_penalty


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
