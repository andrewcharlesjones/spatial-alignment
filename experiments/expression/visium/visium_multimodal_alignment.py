import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from os.path import join as pjoin

from gpsa import VariationalGPSA, matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod, callback_twod_multimodal

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import r2_score

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../../paste")
from src.paste import PASTE, visualization

DATA_DIR = "../../../data/visium/mouse_brain"
SCALEFACTOR = 0.17211704
GRAY_PIXEL_VAL = 0.7
N_GENES = 10

N_SAMPLES = 8000

n_spatial_dims = 2
n_views = 2

m_G = 200
m_X_per_view = 200

N_EPOCHS = 6000
PRINT_EVERY = 50
N_LATENT_GPS = {"expression": None, "histology": None}

device = "cuda" if torch.cuda.is_available() else "cpu"


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


def process_image(adata):

    img_keys = [*adata.uns["spatial"]]
    assert len(img_keys) == 1
    curr_key = img_keys[0]
    img = adata.uns["spatial"][curr_key]["images"]["hires"]
    scale_factor = adata.uns["spatial"][curr_key]["scalefactors"]["tissue_hires_scalef"]
    adata.obsm["spatial"] = adata.obsm["spatial"] * scale_factor

    img_size = img.shape

    x1_img, x2_img = np.meshgrid(
        np.arange(img_size[1]),
        np.arange(img_size[0]),
    )
    X_img = np.vstack([x1_img.ravel(), x2_img.ravel()]).T * 1.0

    pixel_vals = np.array(
        [
            img[X_img[ii, 1].astype(int), X_img[ii, 0].astype(int)]
            for ii in range(X_img.shape[0])
        ]
    )

    nongray_idx = np.where(~np.all(np.round(pixel_vals, 1) == GRAY_PIXEL_VAL, axis=1))[
        0
    ]
    pixel_vals = pixel_vals[nongray_idx, :]
    X_img = X_img[nongray_idx, :]
    X_img_original = X_img.copy()
    pixel_vals_original = pixel_vals.copy()

    ## Remove image pixels outside of expression region
    xmax, ymax = adata.obsm["spatial"].max(0)
    xmin, ymin = adata.obsm["spatial"].min(0)
    inside_idx = np.where(
        (X_img[:, 0] > xmin)
        & (X_img[:, 0] < xmax)
        & (X_img[:, 1] > ymin)
        & (X_img[:, 1] < ymax)
    )[0]
    X_img = X_img[inside_idx]
    pixel_vals = pixel_vals[inside_idx]

    adata.uns["img_spatial"] = X_img
    adata.uns["img_pixels"] = pixel_vals

    return adata


data_slice1 = sc.read_visium(pjoin(DATA_DIR, "sample1"))
data_slice1 = process_data(data_slice1, n_top_genes=6000)
data_slice1 = process_image(data_slice1)


data_slice2 = sc.read_visium(pjoin(DATA_DIR, "sample2"))
data_slice2 = process_data(data_slice2, n_top_genes=6000)
data_slice2 = process_image(data_slice2)


# plt.subplot(121)
# rand_idx = np.random.choice(np.arange(data_slice1.uns["img_spatial"].shape[0]), size=3000, replace=False)
# plt.scatter(
#     data_slice1.uns["img_spatial"][rand_idx][:, 0],
#     data_slice1.uns["img_spatial"][rand_idx][:, 1],
#     c=data_slice1.uns["img_pixels"][rand_idx],
# )
# plt.scatter(
#     data_slice1.obsm["spatial"][:, 0],
#     data_slice1.obsm["spatial"][:, 1],
#     s=6,
#     c=data_slice1.obs.total_counts.values,
# )

# plt.subplot(122)
# rand_idx = np.random.choice(np.arange(data_slice2.uns["img_spatial"].shape[0]), size=3000, replace=False)
# plt.scatter(
#     data_slice2.uns["img_spatial"][rand_idx][:, 0],
#     data_slice2.uns["img_spatial"][rand_idx][:, 1],
#     c=data_slice2.uns["img_pixels"][rand_idx],
# )
# plt.scatter(
#     data_slice2.obsm["spatial"][:, 0],
#     data_slice2.obsm["spatial"][:, 1],
#     s=6,
#     c=data_slice2.obs.total_counts.values,
# )
# plt.show()


data = data_slice1.concatenate(data_slice2)


## Filter for spatially variable genes
shared_gene_names = data.var.gene_ids.index.values
data_knn = data_slice1[:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = np.array(data_knn.X.todense())
nbrs = NearestNeighbors(n_neighbors=2).fit(X_knn)
distances, indices = nbrs.kneighbors(X_knn)

preds = Y_knn[indices[:, 1]]
r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

gene_idx_to_keep = np.where(r2_vals > 0.3)[0]
N_GENES = min(N_GENES, len(gene_idx_to_keep))
gene_names_to_keep = data_knn.var.gene_ids.index.values[gene_idx_to_keep]
gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
if N_GENES < len(gene_names_to_keep):
    gene_names_to_keep = gene_names_to_keep[:N_GENES]
data = data[:, gene_names_to_keep]

## Filter number of samples
if N_SAMPLES is not None:

    if N_SAMPLES < data_slice1.shape[0]:
        rand_idx = np.random.choice(
            np.arange(data_slice1.shape[0]), size=N_SAMPLES, replace=False
        )
        data_slice1 = data_slice1[rand_idx]

    if N_SAMPLES < data_slice1.uns["img_spatial"].shape[0]:
        rand_idx = np.random.choice(
            np.arange(data_slice1.uns["img_spatial"].shape[0]),
            size=N_SAMPLES,
            replace=False,
        )
        data_slice1.uns["img_spatial"] = data_slice1.uns["img_spatial"][rand_idx]
        data_slice1.uns["img_pixels"] = data_slice1.uns["img_pixels"][rand_idx]

    if N_SAMPLES < data_slice2.shape[0]:
        rand_idx = np.random.choice(
            np.arange(data_slice2.shape[0]), size=N_SAMPLES, replace=False
        )
        data_slice2 = data_slice2[rand_idx]

    if N_SAMPLES < data_slice2.uns["img_spatial"].shape[0]:
        rand_idx = np.random.choice(
            np.arange(data_slice2.uns["img_spatial"].shape[0]),
            size=N_SAMPLES,
            replace=False,
        )
        data_slice2.uns["img_spatial"] = data_slice2.uns["img_spatial"][rand_idx]
        data_slice2.uns["img_pixels"] = data_slice2.uns["img_pixels"][rand_idx]

data = data_slice1.concatenate(data_slice2)
data = data[:, gene_names_to_keep]
n_samples_list_expression = [data_slice1.shape[0], data_slice2.shape[0]]
n_samples_list_histology = [
    data_slice1.uns["img_spatial"].shape[0],
    data_slice2.uns["img_spatial"].shape[0],
]
view_idx = [
    np.arange(data_slice1.shape[0]),
    np.arange(data_slice1.shape[0], data_slice1.shape[0] + data_slice2.shape[0]),
]


### Expression
X1_expression = data[data.obs.batch == "0"].obsm["spatial"]
X2_expression = data[data.obs.batch == "1"].obsm["spatial"]
Y1_expression = np.array(data[data.obs.batch == "0"].X.todense())
Y2_expression = np.array(data[data.obs.batch == "1"].X.todense())

X1_expression = scale_spatial_coords(X1_expression)
X2_expression = scale_spatial_coords(X2_expression)

Y1_expression = (Y1_expression - Y1_expression.mean(0)) / Y1_expression.std(0)
Y2_expression = (Y2_expression - Y2_expression.mean(0)) / Y2_expression.std(0)

X_expression = np.concatenate([X1_expression, X2_expression])
Y_expression = np.concatenate([Y1_expression, Y2_expression])

### Histology
X1_histology = data_slice1.uns["img_spatial"]
X2_histology = data_slice2.uns["img_spatial"]
Y1_histology = data_slice1.uns["img_pixels"]
Y2_histology = data_slice2.uns["img_pixels"]

X1_histology = scale_spatial_coords(X1_histology)
X2_histology = scale_spatial_coords(X2_histology)

Y1_histology_rgb = Y1_histology.copy()
Y2_histology_rgb = Y2_histology.copy()


Y1_histology_mean = Y1_histology.mean(0)
Y1_histology_stddev = Y1_histology.std(0)
Y2_histology_mean = Y2_histology.mean(0)
Y2_histology_stddev = Y2_histology.std(0)

Y1_histology = (Y1_histology - Y1_histology_mean) / Y1_histology_stddev
Y2_histology = (Y2_histology - Y2_histology_mean) / Y2_histology_stddev

X_histology = np.concatenate([X1_histology, X2_histology])
Y_histology = np.concatenate([Y1_histology, Y2_histology])
Y_histology_rgb = np.concatenate([Y1_histology_rgb, Y2_histology_rgb])

device = "cuda" if torch.cuda.is_available() else "cpu"

x_expression = torch.from_numpy(X_expression).float().clone()
x_histology = torch.from_numpy(X_histology).float().clone()
y_expression = torch.from_numpy(Y_expression).float().clone()
y_histology = torch.from_numpy(Y_histology).float().clone()
y_histology_rgb = torch.from_numpy(Y_histology_rgb).float().clone()

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
        "outputs": y_histology_rgb,
        "n_samples_list": n_samples_list_histology,
    },
}


model = VariationalGPSA(
    data_dict,
    n_spatial_dims=n_spatial_dims,
    m_X_per_view=m_X_per_view,
    m_G=m_G,
    data_init=True,
    minmax_init=False,
    grid_init=False,
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    # fixed_warp_kernel_variances=np.ones(n_views) * 0.1,
    # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
    n_noise_variance_params=3,
    fixed_view_idx=0,
).to(device)

view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


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

SAVE_DIR = pjoin("out", "multimodal")

pd.DataFrame(X_expression).to_csv(pjoin(SAVE_DIR, "X_expression_visium.csv"))
pd.DataFrame(X_histology).to_csv(pjoin(SAVE_DIR, "X_histology_visium.csv"))

pd.DataFrame(Y_expression).to_csv(pjoin(SAVE_DIR, "Y_expression_visium.csv"))
pd.DataFrame(Y_histology_rgb).to_csv(pjoin(SAVE_DIR, "Y_histology_rgb_visium.csv"))
data.write(pjoin(SAVE_DIR, "data_visium.h5"))

pd.DataFrame(view_idx["expression"]).to_csv(
    pjoin(SAVE_DIR, "view_idx_expression_visium.csv")
)
pd.DataFrame(view_idx["histology"]).to_csv(
    pjoin(SAVE_DIR, "view_idx_histology_visium.csv")
)


for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)

        callback_twod_multimodal(
            model=model,
            data_dict=data_dict_rgb,
            axes=axes_list,
            X_aligned=G_means,
            scatterpoint_size=50,
            rgb=True,
        )

        curr_aligned_coords_expression = G_means["expression"].detach().numpy()
        curr_aligned_coords_histology = G_means["histology"].detach().numpy()

        pd.DataFrame(curr_aligned_coords_expression).to_csv(
            pjoin(SAVE_DIR, "aligned_coords_expression_visium.csv")
        )
        pd.DataFrame(curr_aligned_coords_histology).to_csv(
            pjoin(SAVE_DIR, "aligned_coords_histology_visium.csv")
        )

        plt.savefig(pjoin(SAVE_DIR, "tmp_alignment_multimodal.png"))
        plt.draw()
        plt.pause(1 / 60.0)

plt.close()
