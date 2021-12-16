import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from os.path import join as pjoin
import scanpy as sc
import anndata
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append("../../..")
sys.path.append("../../../data")
from util import (
    compute_size_factors,
    poisson_deviance,
    deviance_feature_selection,
    deviance_residuals,
    pearson_residuals,
)
from util import matern12_kernel, matern32_kernel, rbf_kernel
from models.gpsa_vi_lmc import VariationalWarpGP
from plotting.callbacks import callback_oned, callback_twod

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../../paste")
from src.paste import PASTE, visualization


def scale_spatial_coords(X, max_val=10):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


DATA_DIR = "../../../data/visium/mouse_brain"
N_GENES = 100
N_SAMPLES = 500
N_REPEATS = 10
predict_idx = np.arange(10)
frac_test = 0.2
frac_train = 1 - frac_test

n_spatial_dims = 2
n_views = 2
m_G = 40
m_X_per_view = 40

N_EPOCHS = 2000
PRINT_EVERY = 50
N_LATENT_GPS = {"expression": 5}

spatial_locs_sample1_path = pjoin(
    DATA_DIR, "sample1", "filtered_feature_bc_matrix/spatial_locs_small.csv"
)
data_sample1_path = pjoin(
    DATA_DIR, "sample1", "filtered_feature_bc_matrix/gene_expression_small.csv"
)
X_df_sample1 = pd.read_csv(spatial_locs_sample1_path, index_col=0)
X_orig1 = np.vstack([X_df_sample1.array_col.values, X_df_sample1.array_row.values]).T
X_orig1 = scale_spatial_coords(X_orig1)
slice1 = sc.read_csv(data_sample1_path)
slice1.obsm["spatial"] = X_orig1
sc.pp.filter_genes(slice1, min_counts=15)
sc.pp.filter_cells(slice1, min_counts=100)

spatial_locs_sample2_path = pjoin(
    DATA_DIR, "sample2", "filtered_feature_bc_matrix/spatial_locs_small.csv"
)
data_sample2_path = pjoin(
    DATA_DIR, "sample2", "filtered_feature_bc_matrix/gene_expression_small.csv"
)
X_df_sample2 = pd.read_csv(spatial_locs_sample2_path, index_col=0)
X_orig2 = np.vstack([X_df_sample2.array_col.values, X_df_sample2.array_row.values]).T
X_orig2 = scale_spatial_coords(X_orig2)
slice2 = sc.read_csv(data_sample2_path)
slice2.obsm["spatial"] = X_orig2
sc.pp.filter_genes(slice2, min_counts=15)
sc.pp.filter_cells(slice2, min_counts=100)

errors_union, errors_separate, errors_gpsa = [], [], []

for repeat_idx in range(N_REPEATS):

    rand_idx = np.random.choice(
        np.arange(slice1.shape[0]), size=N_SAMPLES, replace=False
    )
    slice1 = slice1[rand_idx]
    rand_idx = np.random.choice(
        np.arange(slice2.shape[0]), size=N_SAMPLES, replace=False
    )
    slice2 = slice2[rand_idx]

    all_slices = anndata.concat([slice1, slice2])
    n_samples_list = [slice1.shape[0], slice2.shape[0]]
    view_idx = [
        np.arange(slice1.shape[0]),
        np.arange(slice1.shape[0], slice1.shape[0] + slice2.shape[0]),
    ]

    deviances, gene_names = deviance_feature_selection(all_slices.to_df().transpose())
    sorted_idx = np.argsort(-deviances)
    highly_variable_genes = gene_names[sorted_idx][:N_GENES]

    # highly_variable_genes = gene_names[sorted_idx][1:4]
    all_slices = all_slices[:, highly_variable_genes]

    X1 = all_slices.obsm["spatial"][: slice1.shape[0]]
    X2 = all_slices.obsm["spatial"][slice1.shape[0] :]
    Y1_unnormalized = all_slices.X[: slice1.shape[0]]
    Y2_unnormalized = all_slices.X[slice1.shape[0] :]
    Y1 = pearson_residuals(np.array(Y1_unnormalized), theta=100.0)
    Y2 = pearson_residuals(np.array(Y2_unnormalized), theta=100.0)

    Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
    Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

    X = np.concatenate([X1, X2])
    Y = np.concatenate([Y1, Y2])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_outputs = all_slices.shape[1]

    ## Drop part of the second view (this is the part we'll try to predict)
    second_view_idx = view_idx[1]
    n_drop = int(1.0 * view_idx[1].shape[0] // (1 / frac_test))
    test_idx = np.random.choice(second_view_idx, size=n_drop, replace=False)
    keep_idx = np.setdiff1d(second_view_idx, test_idx)

    train_idx = np.concatenate([np.arange(view_idx[0].shape[0]), keep_idx])

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    n_samples_list_train = n_samples_list
    n_samples_list_train[1] -= n_drop

    n_samples_list_test = [[0], [n_drop]]

    X_test = X[test_idx]
    Y_test = Y[test_idx]

    x_train = torch.from_numpy(X_train).float().clone()
    y_train = torch.from_numpy(Y_train).float().clone()
    x_test = torch.from_numpy(X_test).float().clone()
    y_test = torch.from_numpy(Y_test).float().clone()

    data_dict_train = {
        "expression": {
            "spatial_coords": x_train,
            "outputs": y_train,
            "n_samples_list": n_samples_list_train,
        }
    }

    data_dict_test = {
        "expression": {
            "spatial_coords": x_test,
            "outputs": y_test,
            "n_samples_list": n_samples_list_test,
        }
    }

    model = VariationalWarpGP(
        data_dict_train,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=N_LATENT_GPS,
        kernel_func_warp=rbf_kernel,
        kernel_func_data=matern32_kernel,
        mean_function="identity_fixed",
        fixed_warp_kernel_variances=np.ones(n_views) * 0.1,
        fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        # mean_function="identity_initialized",
        # fixed_view_idx=0,
    ).to(device)

    view_idx_train, Ns_train, _, _ = model.create_view_idx_dict(data_dict_train)
    view_idx_test, Ns_test, _, _ = model.create_view_idx_dict(data_dict_test)

    ## Make predictions for naive alignment
    gpr_union = GaussianProcessRegressor(kernel=RBF() + WhiteKernel())

    # gpr_union.fit(X=X_train, y=Y_train[:, predict_idx])
    half_n_samples_train = int((N_SAMPLES * frac_train) // 2)
    gpr_union.fit(
        X=X_train[
            np.concatenate(
                [
                    np.arange(0, half_n_samples_train),
                    np.arange(N_SAMPLES, N_SAMPLES + half_n_samples_train),
                ]
            )
        ],
        y=Y_train[
            np.concatenate(
                [
                    np.arange(0, half_n_samples_train),
                    np.arange(N_SAMPLES, N_SAMPLES + half_n_samples_train),
                ]
            )
        ][:, predict_idx],
    )
    preds = gpr_union.predict(X_test)
    error_union = np.mean(np.sum((preds - Y_test[:, predict_idx]) ** 2, 1))

    print("MSE, union: {}".format(round(error_union, 5)))

    curr_r2 = r2_score(Y_test[:, predict_idx], preds)
    print("R2, union: {}".format(round(curr_r2, 5)), flush=True)

    errors_union.append(curr_r2)

    ## Make predictons for each view separately
    preds, truth = [], []

    for vv in range(n_views):
        gpr_separate = GaussianProcessRegressor(kernel=Matern(nu=1.5) + WhiteKernel())
        curr_trainX = X_train[view_idx_train["expression"][vv]]
        curr_trainY = Y_train[view_idx_train["expression"][vv]]
        curr_testX = X_test[view_idx_test["expression"][vv]]
        curr_testY = Y_test[view_idx_test["expression"][vv]]
        if len(curr_testX) == 0:
            continue
        gpr_separate.fit(X=curr_trainX, y=curr_trainY[:, predict_idx])
        curr_preds = gpr_separate.predict(curr_testX)
        preds.append(curr_preds)
        truth.append(curr_testY[:, predict_idx])

    preds = np.concatenate(preds, axis=0)
    truth = np.concatenate(truth, axis=0)
    error_separate = np.mean(np.sum((preds - truth) ** 2, 1))

    print("MSE, separate: {}".format(round(error_separate, 5)))
    # preds = gpr_union.predict(X_test)

    curr_r2 = r2_score(truth, preds)
    print("R2, separate: {}".format(round(curr_r2, 5)), flush=True)
    errors_separate.append(curr_r2)

    # import ipdb; ipdb.set_trace()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            X_spatial={"expression": x_train},
            view_idx=view_idx_train,
            Ns=Ns_train,
            S=10,
        )

        # Compute loss
        loss = loss_fn(data_dict_train, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), G_means

    # Set up figure.
    # fig = plt.figure(figsize=(14, 7), facecolor="white", constrained_layout=True)
    # ax_dict = fig.subplot_mosaic(
    #     [
    #         ["data", "latent"],
    #     ],
    # )
    # plt.show(block=False)

    for t in range(N_EPOCHS):
        loss, G_means = train(model, model.loss_fn, optimizer)

        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            # print(model.warp_kernel_variances.detach().numpy())

            G_means_test, _, F_samples_test, _, = model.forward(
                X_spatial={"expression": x_test},
                view_idx=view_idx_test,
                Ns=Ns_test,
                prediction_mode=True,
                S=10,
            )

            curr_preds = (
                torch.mean(F_samples_test["expression"], dim=0).detach().numpy()
            )

            # callback_twod(
            #     model,
            #     X_train,
            #     Y_train,
            #     data_expression_ax=ax_dict["data"],
            #     latent_expression_ax=ax_dict["latent"],
            #     # prediction_ax=ax_dict["preds"],
            #     X_aligned=G_means,
            #     # X_test=X_test,
            #     # Y_test_true=Y_test,
            #     # Y_pred=curr_preds,
            #     # X_test_aligned=G_means_test,
            # )
            # plt.draw()
            # plt.pause(1./60)

            # print("R2, GPSA: {}".format(round(r2_score(Y_test, curr_preds), 5)))

            curr_aligned_coords = G_means["expression"].detach().numpy()
            curr_aligned_coords_test = G_means_test["expression"].detach().numpy()

            try:
                gpr_gpsa = GaussianProcessRegressor(kernel=RBF() + WhiteKernel())
                fit_idx = np.concatenate(
                    [
                        np.arange(0, half_n_samples_train),
                        np.arange(N_SAMPLES, N_SAMPLES + half_n_samples_train),
                    ]
                )
                # import ipdb; ipdb.set_trace()
                gpr_gpsa.fit(
                    X=curr_aligned_coords[fit_idx], y=Y_train[fit_idx][:, predict_idx]
                )
                preds = gpr_gpsa.predict(curr_aligned_coords_test)
                # import ipdb; ipdb.set_trace()
                error_gpsa = np.mean(
                    np.sum((preds - Y_test[:, predict_idx]) ** 2, axis=1)
                )
                # print("MSE, GPSA GPR: {}".format(round(error_gpsa, 5)))

                curr_r2 = r2_score(Y_test[:, predict_idx], preds)
                print("R2, GPSA: {}".format(round(curr_r2, 5)), flush=True)
                # print("R2, GPSA: {}".format(round(curr_r2, 5)))
            except:
                continue

    try:
        G_means_test, _, F_samples_test, _, = model.forward(
            X_spatial={"expression": x_test},
            view_idx=view_idx_test,
            Ns=Ns_test,
            prediction_mode=True,
            S=10,
        )

        gpr_gpsa = GaussianProcessRegressor(kernel=RBF() + WhiteKernel())
        gpr_gpsa.fit(X=curr_aligned_coords, y=Y_train)
        preds = gpr_gpsa.predict(curr_aligned_coords_test)
        error_gpsa = np.mean(np.sum((preds - Y_test) ** 2, axis=1))
        print("MSE, GPSA GPR: {}".format(round(error_gpsa, 5)))

        curr_r2 = r2_score(Y_test, preds)
        print("R2, GPSA: {}".format(round(curr_r2, 5)), flush=True)
    except:
        pass

    errors_gpsa.append(curr_r2)

    plt.close()

    results_df = pd.DataFrame(
        {
            "Union": errors_union[: repeat_idx + 1],
            "Separate": errors_separate[: repeat_idx + 1],
            "GPSA": errors_gpsa[: repeat_idx + 1],
        }
    )
    results_df_melted = pd.melt(results_df)
    results_df_melted.to_csv("./out/prediction_comparison_visium.csv")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=results_df_melted, x="variable", y="value", color="gray")
    plt.xlabel("")
    plt.ylabel("R^2")
    plt.title("Visium")
    plt.tight_layout()
    plt.savefig("./out/two_d_prediction_comparison_visium.png")
    # plt.show()
    plt.close()
