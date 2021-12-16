import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import sys

sys.path.append("../..")
from models.gpsa_vi_lmc import VariationalWarpGP

sys.path.append("../../data")
from simulated.generate_oned_data import (
    generate_oned_data_affine_warp,
    generate_oned_data_gp_warp,
)
from plotting.callbacks import callback_oned
from util import ConvergenceChecker


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 30

n_spatial_dims = 1
n_views = 2
n_outputs = 50
n_samples_per_view = 100
m_G = 10
m_X_per_view = 10

N_EPOCHS = 3000
PRINT_EVERY = 25
N_LATENT_GPS = 1
NOISE_VARIANCE = 0.01

# X, Y, n_samples_list, view_idx = generate_oned_data_affine_warp(
#     n_views,
#     n_outputs,
#     n_samples_per_view,
#     noise_variance=NOISE_VARIANCE,
#     n_latent_gps=N_LATENT_GPS,
#     scale_factor=1.,
#     additive_factor=0.3,
# )
X, Y, n_samples_list, view_idx = generate_oned_data_gp_warp(
    n_views,
    n_outputs,
    n_samples_per_view,
    noise_variance=NOISE_VARIANCE,
    n_latent_gps=N_LATENT_GPS,
    kernel_variance=0.25,
    kernel_lengthscale=10.0,
)

## Fit GP on one view to get initial estimates of data kernel parameters
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

kernel = RBF(length_scale=1.0) + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(X[view_idx[0]], Y[view_idx[0]])
data_lengthscales_est = gpr.kernel_.k1.theta[0]

x = torch.from_numpy(X).float().clone()
y = torch.from_numpy(Y).float().clone()

data_dict = {
    "expression": {
        "spatial_coords": x,
        "outputs": y,
        "n_samples_list": n_samples_list,
    }
}

model = VariationalWarpGP(
    data_dict,
    n_spatial_dims=n_spatial_dims,
    m_X_per_view=m_X_per_view,
    m_G=m_G,
    data_init=False,
    minmax_init=False,
    grid_init=True,
    n_latent_gps=N_LATENT_GPS,
    # n_latent_gps=None,
    mean_function="identity_fixed",
    fixed_warp_kernel_variances=np.ones(n_views) * 0.1,
    fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
    fixed_data_kernel_lengthscales=np.exp(gpr.kernel_.k1.theta.astype(np.float32)),
    # mean_function="identity_initialized",
    # fixed_view_idx=0,
).to(device)

view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)


def train(model, loss_fn, optimizer):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        {"expression": x},
        view_idx=view_idx,
        Ns=Ns,
        S=5,
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Set up figure.
fig = plt.figure(figsize=(14, 7), facecolor="white")
data_expression_ax = fig.add_subplot(212, frameon=False)
latent_expression_ax = fig.add_subplot(211, frameon=False)
plt.show(block=False)


convergence_checker = ConvergenceChecker(span=100)

loss_trace = []
error_trace = []
convergence_counter = 0
for t in range(N_EPOCHS):
    loss = train(model, model.loss_fn, optimizer)
    loss_trace.append(loss)
    if t >= convergence_checker.span - 1:
        rel_change = convergence_checker.relative_change(loss_trace)
        is_converged = convergence_checker.converged(loss_trace, tol=1e-4)
        if is_converged:
            convergence_counter += 1

            if convergence_counter == 2:
                print("CONVERGED")
                break

        else:
            convergence_counter = 0

        # import ipdb; ipdb.set_trace()
    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            {"expression": x}, view_idx=view_idx, Ns=Ns, S=3
        )
        callback_oned(
            model,
            X,
            Y=Y,
            # F_samples=F_latent_samples["expression"].mean(0),
            X_aligned=G_means,
            data_expression_ax=data_expression_ax,
            latent_expression_ax=latent_expression_ax,
        )
        # import ipdb; ipdb.set_trace()
        # plt.scatter(G_means["expression"].detach().numpy().squeeze()[:200], G_means["expression"].detach().numpy().squeeze()[200:])
        err = np.mean(
            (
                G_means["expression"].detach().numpy().squeeze()[:n_samples_per_view]
                - G_means["expression"].detach().numpy().squeeze()[n_samples_per_view:]
            )
            ** 2
        )
        print("Error: {}".format(err))
        error_trace.append(loss)
        if t >= convergence_checker.span - 1:
            print(rel_change)

print("Done!")

plt.close()

import matplotlib

font = {"size": LATEX_FONTSIZE}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(10, 10))
data_expression_ax = fig.add_subplot(211, frameon=False)
latent_expression_ax = fig.add_subplot(212, frameon=False)

callback_oned(
    model,
    X,
    Y=Y,
    # F_samples=F_latent_samples["expression"].mean(0),
    X_aligned=G_means,
    data_expression_ax=data_expression_ax,
    latent_expression_ax=latent_expression_ax,
)
# callback_oned(model, X, Y, data_expression_ax, latent_expression_ax)

plt.tight_layout()
plt.savefig("../../plots/one_d_simulation.png")
plt.show()

import ipdb

ipdb.set_trace()
