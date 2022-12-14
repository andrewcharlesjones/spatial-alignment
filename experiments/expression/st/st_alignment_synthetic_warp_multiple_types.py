import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys

from gpsa import VariationalGPSA, rbf_kernel
from gpsa.plotting import callback_twod

sys.path.append("../../..")
sys.path.append("../../../data/st")
from load_st_data import load_st_data

sys.path.append("../../../data")
from warps import apply_gp_warp, apply_linear_warp, apply_polar_warp


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

## For PASTE
import scanpy as sc
import squidpy as sq
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../paste")
from src.paste import PASTE, visualization

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


N_LAYERS = 1
N_REPEATS = 10
n_spatial_dims = 2
n_views = 2


m_G = 40
m_X_per_view = 40

N_EPOCHS = 3_000
PRINT_EVERY = 200
N_LATENT_GPS = {"expression": None}
NOISE_VARIANCE = 0.0


device = "cuda" if torch.cuda.is_available() else "cpu"


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=100)
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    # sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


data_slice1 = load_st_data(layers=np.arange(N_LAYERS) + 1)[0]
process_data(data_slice1, n_top_genes=3000)

sq.gr.spatial_neighbors(data_slice1)
sq.gr.spatial_autocorr(
    data_slice1,
    mode="moran",
)


moran_scores = data_slice1.uns["moranI"]
genes_to_keep = moran_scores.index.values[np.where(moran_scores.I.values > 0.3)[0]]
N_GENES = len(genes_to_keep)
data_slice1 = data_slice1[:, genes_to_keep]


X_orig = data_slice1.obsm["spatial"]
Y_orig = data_slice1.X
n_samples_per_view = len(X_orig)

X_orig = scale_spatial_coords(X_orig)


errors_gpsa_gpwarp = np.zeros(N_REPEATS)
errors_paste_gpwarp = np.zeros(N_REPEATS)

errors_gpsa_linearwarp = np.zeros(N_REPEATS)
errors_paste_linearwarp = np.zeros(N_REPEATS)

errors_gpsa_polarwarp = np.zeros(N_REPEATS)
errors_paste_polarwarp = np.zeros(N_REPEATS)

for rep_ii in range(N_REPEATS):

    ## GP warp

    X, Y, n_samples_list, view_idx = apply_gp_warp(
        X_orig,
        Y_orig,
        n_views=2,
        kernel_variance=0.5,
        kernel_lengthscale=5,
        noise_variance=0.0,
    )

    X[:n_samples_per_view] = X_orig

    n_outputs = Y.shape[1]

    ##  PASTE
    slice1 = anndata.AnnData(Y[view_idx[0]])
    slice2 = anndata.AnnData(Y[view_idx[1]])

    slice1.obsm["spatial"] = X[view_idx[0]]
    slice2.obsm["spatial"] = X[view_idx[1]]

    pi12 = PASTE.pairwise_align(slice1, slice2, alpha=0.1)

    slices = [slice1, slice2]
    pis = [pi12]
    new_slices = visualization.stack_slices_pairwise(slices, pis)

    err_paste = np.mean(
        np.sum(
            (new_slices[0].obsm["spatial"] - new_slices[1].obsm["spatial"]) ** 2, axis=1
        )
    )
    errors_paste_gpwarp[rep_ii] = err_paste
    print("Error PASTE: {}".format(err_paste))

    # Standardize expression
    Y = (Y - Y.mean(0)) / Y.std(0)

    ## Drop part of the second view (this is the part we'll try to predict)
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
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=N_LATENT_GPS,
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
        # fixed_warp_kernel_variances=np.ones(n_views) * 1.,
        # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        fixed_view_idx=0,
    ).to(device)

    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns
        )

        # Compute loss
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), G_means

    # fig = plt.figure(figsize=(14, 7), facecolor="white", constrained_layout=True)
    # data_expression_ax = fig.add_subplot(121, frameon=False)
    # latent_expression_ax = fig.add_subplot(122, frameon=False)
    for t in range(N_EPOCHS):
        loss, G_means = train(model, model.loss_fn, optimizer)

        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))

            # callback_twod(
            #     model,
            #     X,
            #     Y,
            #     data_expression_ax=data_expression_ax,
            #     latent_expression_ax=latent_expression_ax,
            #     # prediction_ax=ax_dict["preds"],
            #     X_aligned=G_means,
            #     # X_test=X_test,
            #     # Y_test_true=Y_test,
            #     # Y_pred=curr_preds,
            #     # X_test_aligned=G_means_test,
            # )
            # plt.draw()
            # plt.pause(1 / 60.0)

            err_gpsa = np.mean(
                np.sum(
                    (
                        G_means["expression"]
                        .detach()
                        .numpy()
                        .squeeze()[:n_samples_per_view]
                        - G_means["expression"]
                        .detach()
                        .numpy()
                        .squeeze()[n_samples_per_view:]
                    )
                    ** 2,
                    axis=1,
                )
            )
            print("Error: {}".format(err_gpsa))

    errors_gpsa_gpwarp[rep_ii] = err_gpsa

    plt.close()

    gp_warp_results = pd.melt(
        pd.DataFrame(
            {
                "PASTE": errors_paste_gpwarp[: rep_ii + 1],
                "GPSA": errors_gpsa_gpwarp[: rep_ii + 1],
            }
        )
    )

    ########### Linear warp

    X, Y, n_samples_list, view_idx = apply_linear_warp(
        X_orig,
        Y_orig,
        n_views=2,
        linear_slope_variance=0.05,
        linear_intercept_variance=0.05,
        noise_variance=NOISE_VARIANCE,
        rotation=False,
    )

    X[:n_samples_per_view] = X_orig

    n_outputs = Y.shape[1]

    ##  PASTE
    slice1 = anndata.AnnData(Y[view_idx[0]])
    slice2 = anndata.AnnData(Y[view_idx[1]])

    slice1.obsm["spatial"] = X[view_idx[0]]
    slice2.obsm["spatial"] = X[view_idx[1]]

    # import ipdb; ipdb.set_trace()
    pi12 = PASTE.pairwise_align(slice1, slice2, alpha=0.1)

    slices = [slice1, slice2]
    pis = [pi12]
    new_slices = visualization.stack_slices_pairwise(slices, pis)

    err_paste = np.mean(
        np.sum(
            (new_slices[0].obsm["spatial"] - new_slices[1].obsm["spatial"]) ** 2, axis=1
        )
    )
    errors_paste_linearwarp[rep_ii] = err_paste
    print("Error PASTE: {}".format(err_paste))

    # Standardize expression
    Y = (Y - Y.mean(0)) / Y.std(0)

    ## Drop part of the second view (this is the part we'll try to predict)
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
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=N_LATENT_GPS,
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
        # fixed_warp_kernel_variances=np.ones(n_views) * 1.,
        # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        fixed_view_idx=0,
    ).to(device)

    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns
        )

        # Compute loss
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), G_means

    # fig = plt.figure(figsize=(14, 7), facecolor="white", constrained_layout=True)
    # data_expression_ax = fig.add_subplot(121, frameon=False)
    # latent_expression_ax = fig.add_subplot(122, frameon=False)
    for t in range(N_EPOCHS):
        loss, G_means = train(model, model.loss_fn, optimizer)

        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))

            # callback_twod(
            #     model,
            #     X,
            #     Y,
            #     data_expression_ax=data_expression_ax,
            #     latent_expression_ax=latent_expression_ax,
            #     # prediction_ax=ax_dict["preds"],
            #     X_aligned=G_means,
            #     # X_test=X_test,
            #     # Y_test_true=Y_test,
            #     # Y_pred=curr_preds,
            #     # X_test_aligned=G_means_test,
            # )
            # plt.draw()
            # plt.pause(1 / 60.0)

            err_gpsa = np.mean(
                np.sum(
                    (
                        G_means["expression"]
                        .detach()
                        .numpy()
                        .squeeze()[:n_samples_per_view]
                        - G_means["expression"]
                        .detach()
                        .numpy()
                        .squeeze()[n_samples_per_view:]
                    )
                    ** 2,
                    axis=1,
                )
            )
            print("Error: {}".format(err_gpsa))
    errors_gpsa_linearwarp[rep_ii] = err_gpsa

    plt.close()

    linear_warp_results = pd.melt(
        pd.DataFrame(
            {
                "PASTE": errors_paste_linearwarp[: rep_ii + 1],
                "GPSA": errors_gpsa_linearwarp[: rep_ii + 1],
            }
        )
    )

    ########### Polar warp

    X, Y, n_samples_list, view_idx = apply_polar_warp(
        X_orig,
        Y_orig,
        n_views=2,
        linear_slope_variance=0.1,
        linear_intercept_variance=0.1,
        noise_variance=NOISE_VARIANCE,
        rotation=False,
    )

    X[:n_samples_per_view] = X_orig

    n_outputs = Y.shape[1]

    ##  PASTE
    slice1 = anndata.AnnData(Y[view_idx[0]])
    slice2 = anndata.AnnData(Y[view_idx[1]])

    slice1.obsm["spatial"] = X[view_idx[0]]
    slice2.obsm["spatial"] = X[view_idx[1]]

    # import ipdb; ipdb.set_trace()
    pi12 = PASTE.pairwise_align(slice1, slice2, alpha=0.1)

    slices = [slice1, slice2]
    pis = [pi12]
    new_slices = visualization.stack_slices_pairwise(slices, pis)

    err_paste = np.mean(
        np.sum(
            (new_slices[0].obsm["spatial"] - new_slices[1].obsm["spatial"]) ** 2, axis=1
        )
    )
    errors_paste_polarwarp[rep_ii] = err_paste
    print("Error PASTE: {}".format(err_paste))

    # Standardize expression
    Y = (Y - Y.mean(0)) / Y.std(0)

    ## Drop part of the second view (this is the part we'll try to predict)
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
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=N_LATENT_GPS,
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
        # fixed_warp_kernel_variances=np.ones(n_views) * 1.,
        # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        fixed_view_idx=0,
    ).to(device)

    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns
        )

        # Compute loss
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), G_means

    # fig = plt.figure(figsize=(14, 7), facecolor="white", constrained_layout=True)
    # data_expression_ax = fig.add_subplot(121, frameon=False)
    # latent_expression_ax = fig.add_subplot(122, frameon=False)
    for t in range(N_EPOCHS):
        loss, G_means = train(model, model.loss_fn, optimizer)

        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))

            # callback_twod(
            #     model,
            #     X,
            #     Y,
            #     data_expression_ax=data_expression_ax,
            #     latent_expression_ax=latent_expression_ax,
            #     # prediction_ax=ax_dict["preds"],
            #     X_aligned=G_means,
            #     # X_test=X_test,
            #     # Y_test_true=Y_test,
            #     # Y_pred=curr_preds,
            #     # X_test_aligned=G_means_test,
            # )
            # plt.draw()
            # plt.pause(1 / 60.0)

            err_gpsa = np.mean(
                np.sum(
                    (
                        G_means["expression"]
                        .detach()
                        .numpy()
                        .squeeze()[:n_samples_per_view]
                        - G_means["expression"]
                        .detach()
                        .numpy()
                        .squeeze()[n_samples_per_view:]
                    )
                    ** 2,
                    axis=1,
                )
            )
            print("Error: {}".format(err_gpsa))
    errors_gpsa_polarwarp[rep_ii] = err_gpsa

    plt.close()

    polar_warp_results = pd.melt(
        pd.DataFrame(
            {
                "PASTE": errors_paste_polarwarp[: rep_ii + 1],
                "GPSA": errors_gpsa_polarwarp[: rep_ii + 1],
            }
        )
    )

    gp_warp_results["Warp type"] = "GP"
    linear_warp_results["Warp type"] = "Linear"
    polar_warp_results["Warp type"] = "Polar"

    results_df = pd.concat(
        [gp_warp_results, linear_warp_results, polar_warp_results], axis=0
    )

    results_df.to_csv("./out/st_alignment_synthetic_warp_mulitple_types.csv")

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=results_df, x="Warp type", y="value", hue="variable")
    plt.ylabel("Error")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("./out/st_alignment_synthetic_warp_mulitple_types.png")
    # plt.show()
    plt.close()

import ipdb

ipdb.set_trace()
