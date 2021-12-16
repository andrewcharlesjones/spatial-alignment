import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from os.path import join as pjoin

sys.path.append("../../..")
sys.path.append("../../../data")
from warps import apply_gp_warp_multimodal
from models.gpsa_vi_lmc import VariationalWarpGP
from plotting.callbacks import callback_twod_multimodal, callback_twod

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../../paste")
from src.paste import PASTE, visualization

DATA_DIR = "../../../data/visium/mouse_brain/sample1"
SCALEFACTOR = 0.17211704
GRAY_PIXEL_VAL = 0.7
N_GENES = 10

N_SAMPLES = 2000

device = "cuda" if torch.cuda.is_available() else "cpu"

## Load data
spatial_locs_sample1_path = pjoin(
    DATA_DIR, "filtered_feature_bc_matrix/spatial_locs_small.csv"
)
data_sample1_path = pjoin(
    DATA_DIR, "filtered_feature_bc_matrix/gene_expression_small.csv"
)
X_df_sample1 = pd.read_csv(spatial_locs_sample1_path, index_col=0)
Y_df_sample1 = pd.read_csv(data_sample1_path, index_col=0)
X_df_sample1["x_position"] = X_df_sample1.row_pxl * SCALEFACTOR
X_df_sample1["y_position"] = X_df_sample1.col_pxl * SCALEFACTOR

# import ipdb; ipdb.set_trace()

## Load histology image
tissue_image_hires_sample1 = plt.imread(
    pjoin(DATA_DIR, "spatial/tissue_hires_image.png")
)

## Make box around histology image part that has expression reads
image_xlims = np.array(
    [int(X_df_sample1.x_position.min()), int(X_df_sample1.x_position.max())]
)
image_ylims = np.array(
    [int(X_df_sample1.y_position.min()), int(X_df_sample1.y_position.max())]
)
tissue_img_cropped_sample1 = tissue_image_hires_sample1[
    image_ylims[0] : image_ylims[1], image_xlims[0] : image_xlims[1]
]

## Convert image pixel coordinates into (x, y) coordinates
x1_img, x2_img = np.meshgrid(
    np.arange(tissue_img_cropped_sample1.shape[1]),
    np.arange(tissue_img_cropped_sample1.shape[0]),
)
X_img = np.vstack([x1_img.ravel(), x2_img.ravel()]).T

## Adjust expression spatial coordinates accordingly
X_df_sample1["x_position"] -= image_xlims[0]
X_df_sample1["y_position"] -= image_ylims[0]

pixel_vals = np.array(
    [
        tissue_img_cropped_sample1[X_img[ii, 1], X_img[ii, 0]]
        for ii in range(X_img.shape[0])
    ]
)

## Remove gray border on image
nongray_idx = np.where(~np.all(np.round(pixel_vals, 1) == GRAY_PIXEL_VAL, axis=1))[0]
pixel_vals = pixel_vals[nongray_idx, :]
X_img = X_img[nongray_idx, :]
X_img_original = X_img.copy()
pixel_vals_original = pixel_vals.copy()

## Subsample pixels
rand_idx = np.random.choice(np.arange(X_img.shape[0]), size=N_SAMPLES, replace=False)
X_img = X_img[rand_idx, :]
pixel_vals = pixel_vals[rand_idx, :]

##### Prep expression #####
X_orig_sample1 = X_df_sample1[["x_position", "y_position"]]

X1_expression = X_orig_sample1.values

assert np.all(X_df_sample1.index.values == Y_df_sample1.index.values)

## Select high-variance genes (columns are already sorted here)
chosen_idx = np.arange(N_GENES)
gene_names = Y_df_sample1.columns.values[chosen_idx]
Y_orig_unstdized_sample1 = Y_df_sample1.values[:, chosen_idx]

assert X_orig_sample1.shape[0] == Y_orig_unstdized_sample1.shape[0]

## Subsample expression locations
n1_expression = X1_expression.shape[0]
expression1_subsample_idx = np.random.choice(
    np.arange(n1_expression), size=N_SAMPLES, replace=False
)
X1_expression = X1_expression[expression1_subsample_idx]
Y1_expression = Y_orig_unstdized_sample1[expression1_subsample_idx]

## Standardize expression
Y1_expression = (Y1_expression - Y1_expression.mean(0)) / Y1_expression.std(0)

##### Histology #####
X1_histology = X_img
Y1_histology = pixel_vals

## Standardize histology
Y_histology_mean = Y1_histology.mean(0)
Y_histology_stddev = Y1_histology.std(0)
Y1_histology = (Y1_histology - Y_histology_mean) / Y_histology_stddev

assert X1_histology.shape[0] == Y1_histology.shape[0]


X, Y, n_samples_list, view_idx = apply_gp_warp_multimodal(
    [X1_expression, X1_histology],
    [Y1_expression, Y1_histology],
    n_views=2,
    kernel_variance=0.25,
    kernel_lengthscale=10,
    noise_variance=0.0,
)

X_expression, X_histology = X
Y_expression, Y_histology = Y
n_samples_list_expression, n_samples_list_histology = n_samples_list


n_spatial_dims = 2
n_views = 2

m_G = 40
m_X_per_view = 40

N_EPOCHS = 3000
PRINT_EVERY = 50
N_LATENT_GPS = {"expression": 5, "histology": None}
NOISE_VARIANCE = 0.0


import matplotlib

matplotlib.rcParams["text.usetex"] = False

x_expression = torch.from_numpy(X_expression).float().clone()
x_histology = torch.from_numpy(X_histology).float().clone()
y_expression = torch.from_numpy(Y_expression).float().clone()
y_histology = torch.from_numpy(Y_histology).float().clone()


data_dict = {
    "expression": {
        "spatial_coords": x_expression,
        "outputs": y_expression,
        "n_samples_list": n_samples_list_expression,
    },
    "histology": {
        "spatial_coords": x_histology,
        "outputs": y_histology,
        "n_samples_list": n_samples_list_histology,
    },
}


data_dict_rgb = {
    "expression": {
        "spatial_coords": x_expression,
        "outputs": y_expression,
        "n_samples_list": n_samples_list_expression,
    },
    "histology": {
        "spatial_coords": x_histology,
        "outputs": y_histology * Y_histology_stddev + Y_histology_mean,
        "n_samples_list": n_samples_list_histology,
    },
}

# import ipdb; ipdb.set_trace()


model = VariationalWarpGP(
    data_dict,
    n_spatial_dims=n_spatial_dims,
    m_X_per_view=m_X_per_view,
    m_G=m_G,
    data_init=True,
    minmax_init=False,
    grid_init=False,
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    fixed_warp_kernel_variances=np.ones(n_views) * 0.1,
    fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
    n_noise_variance_params=3,
    # mean_function="identity_initialized",
    # fixed_view_idx=0,
).to(device)

view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)


def train(model, loss_fn, optimizer):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        X_spatial={"expression": x_expression, "histology": x_histology},
        view_idx=view_idx,
        Ns=Ns,
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), G_means


# Set up figure.
fig = plt.figure(figsize=(7, 7), facecolor="white", constrained_layout=True)
# ax_dict = fig.subplot_mosaic(
#     [
#         ["data", "latent"],
#     ],
# )
data_expression_ax = fig.add_subplot(221, frameon=False)
data_histology_ax = fig.add_subplot(222, frameon=False)
latent_expression_ax = fig.add_subplot(223, frameon=False)
latent_histology_ax = fig.add_subplot(224, frameon=False)
axes_list = [
    data_expression_ax,
    data_histology_ax,
    latent_expression_ax,
    latent_histology_ax,
]

plt.show(block=False)

for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
        # print(model.warp_kernel_variances.detach().numpy())

        # G_means_test, _, F_samples_test, _, = model.forward(
        #     X_spatial={"expression": x_test},
        #     view_idx=view_idx_test,
        #     Ns=Ns_test,
        #     prediction_mode=True,
        #     S=10,
        # )

        # curr_preds = torch.mean(F_samples_test["expression"], dim=0)

        callback_twod_multimodal(
            model=model,
            data_dict=data_dict_rgb,
            axes=axes_list,
            X_aligned=G_means,
            scatterpoint_size=50,
            rgb=True,
        )

        plt.savefig("./out/tmp_alignment_multimodal.png")
        plt.draw()
        plt.pause(1 / 60.0)

        err = np.mean(
            (
                G_means["expression"].detach().numpy().squeeze()[:N_SAMPLES]
                - G_means["expression"].detach().numpy().squeeze()[N_SAMPLES:]
            )
            ** 2
        )
        print("Error: {}".format(err))

plt.close()

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(14, 14))
# data_expression_ax = fig.add_subplot(121, frameon=False)
# latent_expression_ax = fig.add_subplot(122, frameon=False)

data_expression_ax = fig.add_subplot(221, frameon=False)
data_histology_ax = fig.add_subplot(222, frameon=False)
latent_expression_ax = fig.add_subplot(223, frameon=False)
latent_histology_ax = fig.add_subplot(224, frameon=False)
axes_list = [
    data_expression_ax,
    data_histology_ax,
    latent_expression_ax,
    latent_histology_ax,
]

# import ipdb; ipdb.set_trace()
data_dict["histology"]["outputs"] = (
    data_dict["histology"]["outputs"] * Y_histology_stddev
)
data_dict["histology"]["outputs"] = data_dict["histology"]["outputs"] + Y_histology_mean

callback_twod_multimodal(
    model=model,
    data_dict=data_dict,
    axes=axes_list,
    X_aligned=G_means,
    rgb=True,
)
# latent_expression_ax.set_title("Aligned data, GPSA")
latent_expression_ax.set_axis_off()
data_expression_ax.set_axis_off()
latent_histology_ax.set_axis_off()
data_histology_ax.set_axis_off()
# plt.axis("off")

plt.tight_layout()
plt.savefig("./out/visium_multimodal_alignment.png")
# plt.show()
plt.close()
