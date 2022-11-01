import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
from os.path import join as pjoin
import pandas as pd

from gpsa import VariationalGPSA
from gpsa import matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod

device = "cuda" if torch.cuda.is_available() else "cpu"

N_SPATIAL_DIMS = 2
N_VIEWS = 2
M_G = 25
M_X_PER_VIEW = 25
N_OUTPUTS = 5
FIXED_VIEW_IDX = 0
N_LATENT_GPS = {"expression": None}

N_EPOCHS = 3000
PRINT_EVERY = 100


DATA_DIR = "../../../data/ben"
data1 = pd.read_csv(pjoin(DATA_DIR, "cell_table_immune.csv"))
data2 = pd.read_csv(pjoin(DATA_DIR, "cell_table_tumor.csv"))

# Spatial coordinates for immune cells
X1 = data1[["centroid-0", "centroid-1"]].values

# Spatial coordinates for tumor cells
X2 = data2[["centroid-0", "centroid-1"]].values

# Get overlapping marker names
marker_col_idx = np.arange(1, 47)
overlapping_markers = np.intersect1d(
    data1.columns.values[marker_col_idx], data2.columns.values[marker_col_idx]
)

# Marker data for immune cells
Y1 = data1[overlapping_markers].values

# Marker data for tumor cells
Y2 = data2[overlapping_markers].values

# Make sure spatial coordinates and marker data have same number of cells
assert len(X1) == len(Y1)
assert len(X2) == len(Y2)

# Concatenate across batches
X = np.vstack([X1, X2])
Y = np.vstack([Y1, Y2])

# Make array of batch indicators (0 or 1)
batch_label = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])

# List of cell indices for each batch
view_idx = [np.where(batch_label == ii)[0] for ii in range(2)]

# Number of samples in each batch
n_samples_list = [len(x) for x in view_idx]

x = torch.from_numpy(X).float().clone()
y = torch.from_numpy(Y).float().clone()

data_dict = {
    "expression": {
        "spatial_coords": x,
        "outputs": y,
        "n_samples_list": n_samples_list,
    }
}


model = VariationalGPSA(
    data_dict,
    n_spatial_dims=N_SPATIAL_DIMS,
    m_X_per_view=M_X_PER_VIEW,
    m_G=M_G,
    data_init=True,
    minmax_init=False,
    grid_init=False,
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    kernel_func_warp=rbf_kernel,
    kernel_func_data=rbf_kernel,
    fixed_view_idx=FIXED_VIEW_IDX,
).to(device)

view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train(model, loss_fn, optimizer):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        {"expression": x}, view_idx=view_idx, Ns=Ns, S=5
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# Set up figure.
fig = plt.figure(figsize=(14, 7), facecolor="white", constrained_layout=True)
data_expression_ax = fig.add_subplot(121, frameon=False)
latent_expression_ax = fig.add_subplot(122, frameon=False)
plt.show(block=False)

for t in range(N_EPOCHS):
    loss = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
        G_means, _, _, _ = model.forward({"expression": x}, view_idx=view_idx, Ns=Ns)

        callback_twod(
            model,
            X,
            Y,
            data_expression_ax=data_expression_ax,
            latent_expression_ax=latent_expression_ax,
            X_aligned=G_means,
            s=600,
        )
        plt.draw()
        plt.pause(1 / 60.0)

print("Done!")

plt.close()
