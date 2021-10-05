import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import sys

sys.path.append("../..")
from models.gpsa_vi_lmc import VariationalWarpGP
from data.simulated.generate_twod_data import generate_twod_data
from plotting.callbacks import callback_twod


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 50

n_spatial_dims = 1
n_views = 2
n_outputs = 5
n_samples_per_view = 100
m_G = 9
m_X_per_view = 9

N_EPOCHS = 3000
PRINT_EVERY = 50
N_LATENT_GPS = 2

X, Y, n_samples_list, view_idx = generate_twod_data(
    n_views, n_outputs, grid_size=15, n_latent_gps=N_LATENT_GPS
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
    # n_latent_gps=n_latent_gps,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train(model, loss_fn, optimizer):
    model.train()

    # Forward pass
    G_samples, F_samples = model.forward({"expression": x})

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Set up figure.
fig = plt.figure(figsize=(14, 7), facecolor="white")
data_expression_ax = fig.add_subplot(121, frameon=False)
latent_expression_ax = fig.add_subplot(122, frameon=False)
plt.show(block=False)

loss_trace = []
error_trace = []
for t in range(N_EPOCHS):
    loss = train(model, model.loss_fn, optimizer)
    loss_trace.append(loss)
    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
        F_latent_samples = model.forward({"expression": x})
        callback_twod(model, X, Y, data_expression_ax, latent_expression_ax)
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
callback_twod(model, X, Y, data_expression_ax, latent_expression_ax)

plt.tight_layout()
plt.savefig("../../plots/two_d_simulation.png")
plt.show()

import ipdb

ipdb.set_trace()
