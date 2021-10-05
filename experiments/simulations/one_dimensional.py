import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import sys

sys.path.append("../..")
from models.gpsa_vi_lmc import VariationalWarpGP
from data.simulated.generate_oned_data import (
    generate_oned_data_affine_warp,
    generate_oned_data_gp_warp,
)
from plotting.callbacks import callback_oned


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 50

n_spatial_dims = 1
n_views = 2
n_outputs = 2
n_samples_per_view = 100
m_G = 9
m_X_per_view = 9

N_EPOCHS = 3000
PRINT_EVERY = 25
N_LATENT_GPS = 1
NOISE_VARIANCE = 0.0

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
    kernel_variance=0.5,
    kernel_lengthscale=5.0,
)

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
    mean_function="identity_fixed",
    # fixed_warp_kernel_variances=np.ones(n_views) * 0.5,
    # fixed_warp_kernel_lengthscales=np.ones(n_views) * 5,
    # mean_function="identity_initialized",
    fixed_view_idx=1,
).to(device)

view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train(model, loss_fn, optimizer):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        {"expression": x}, view_idx=view_idx, Ns=Ns, S=2,
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Set up figure.
fig = plt.figure(figsize=(15, 7), facecolor="white")
data_expression_ax = fig.add_subplot(211, frameon=False)
latent_expression_ax = fig.add_subplot(212, frameon=False)
plt.show(block=False)

loss_trace = []
error_trace = []
for t in range(N_EPOCHS):
    loss = train(model, model.loss_fn, optimizer)
    loss_trace.append(loss)
    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
        print(np.exp(model.warp_kernel_variances.detach().numpy()))
        print(np.exp(model.warp_kernel_lengthscales.detach().numpy()))
        # print(model.W_dict["expression"])
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            {"expression": x}, view_idx=view_idx, Ns=Ns, S=1
        )
        # import ipdb; ipdb.set_trace()
        callback_oned(
            model,
            X,
            Y=Y,
            # F_samples=None,
            # F_samples=F_latent_samples["expression"].mean(0),
            X_aligned=G_means,
            # X_aligned={"expression": G_samples["expression"].mean(0)},
            # X_aligned={"expression": F_samples["expression"].squeeze()},
            data_expression_ax=data_expression_ax,
            latent_expression_ax=latent_expression_ax,
        )
        error_trace.append(loss)

print("Done!")

plt.close()

import matplotlib

font = {"size": LATEX_FONTSIZE}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(10, 10))
data_expression_ax = fig.add_subplot(211, frameon=False)
latent_expression_ax = fig.add_subplot(212, frameon=False)
callback_oned(model, X, Y, data_expression_ax, latent_expression_ax)

plt.tight_layout()
plt.savefig("../../plots/one_d_simulation.png")
plt.show()

import ipdb

ipdb.set_trace()
