import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import seaborn as sns
import sys

from gpsa import VariationalGPSA

sys.path.append("../../data")
from simulated.generate_twod_data import (
    generate_twod_data,
)
from gpsa.plotting import callback_twod

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

import matplotlib

LATEX_FONTSIZE = 30

font = {"size": LATEX_FONTSIZE}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


device = "cuda" if torch.cuda.is_available() else "cpu"


n_spatial_dims = 2
n_views = 2
n_outputs = 10
m_G = 20
m_X_per_view = 20

N_EPOCHS = 1000
PRINT_EVERY = 100
N_LATENT_GPS = {"expression": 3}
NOISE_VARIANCE = 0.1

FRAC_TEST = 0.2

N_REPEATS = 2

errors_union, errors_separate, errors_gpsa = [], [], []

for repeat_idx in range(N_REPEATS):

    X, Y, n_samples_list, view_idx = generate_twod_data(
        n_views,
        n_outputs,
        grid_size=20,
        n_latent_gps=N_LATENT_GPS["expression"],
        kernel_lengthscale=10.0,
        kernel_variance=0.5,
    )
    X -= X.min(0)
    X /= X.max(0)
    X *= 10
    n_samples_per_view = X.shape[0] // n_views

    assert np.array_equal(Y[:n_samples_per_view], Y[n_samples_per_view:])

    ## Drop part of the second view (this is the part we'll try to predict)
    second_view_idx = view_idx[1]
    n_drop = int(1.0 * n_samples_per_view * FRAC_TEST)
    test_idx = np.random.choice(second_view_idx, size=n_drop, replace=False)
    keep_idx = np.setdiff1d(second_view_idx, test_idx)

    train_idx = np.concatenate([np.arange(n_samples_per_view), keep_idx])

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

    model = VariationalGPSA(
        data_dict_train,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=N_LATENT_GPS,
        mean_function="identity_fixed",
        fixed_warp_kernel_variances=np.ones(n_views) * 0.25,
        fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        # mean_function="identity_initialized",
        # fixed_view_idx=0,
    ).to(device)

    view_idx_train, Ns_train, _, _ = model.create_view_idx_dict(data_dict_train)
    view_idx_test, Ns_test, _, _ = model.create_view_idx_dict(data_dict_test)

    ## Make predictions for naive alignment
    gpr_union = GaussianProcessRegressor(kernel=RBF() + WhiteKernel())
    gpr_union.fit(X=X_train, y=Y_train)
    preds = gpr_union.predict(X_test)
    error_union = np.mean(np.sum((preds - Y_test) ** 2, axis=1))
    errors_union.append(error_union)
    print("MSE, union: {}".format(round(error_union, 5)))
    # r2_union = r2_score(Y_test, preds)
    # print("R2, union: {}".format(round(r2_union, 5)))


    ## Make predictons for each view separately
    preds, truth = [], []

    for vv in range(n_views):
        gpr_separate = GaussianProcessRegressor(kernel=RBF() + WhiteKernel())
        curr_trainX = X_train[view_idx_train["expression"][vv]]
        curr_trainY = Y_train[view_idx_train["expression"][vv]]
        curr_testX = X_test[view_idx_test["expression"][vv]]
        curr_testY = Y_test[view_idx_test["expression"][vv]]
        if len(curr_testX) == 0:
            continue
        gpr_separate.fit(X=curr_trainX, y=curr_trainY)
        curr_preds = gpr_separate.predict(curr_testX)
        preds.append(curr_preds)
        truth.append(curr_testY)

    preds = np.concatenate(preds, axis=0)
    truth = np.concatenate(truth, axis=0)
    error_separate = np.mean(np.sum((preds - truth) ** 2, axis=1))
    errors_separate.append(error_separate)
    print("MSE, separate: {}".format(round(error_separate, 5)))
    # r2_sep = r2_score(truth, preds)
    # print("R2, sep: {}".format(round(r2_sep, 5)))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            X_spatial={"expression": x_train}, view_idx=view_idx_train, Ns=Ns_train
        )

        # Compute loss
        loss = loss_fn(data_dict_train, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), G_means

    # Set up figure.
    fig = plt.figure(figsize=(18, 7), facecolor="white", constrained_layout=True)
    data_expression_ax = fig.add_subplot(131, frameon=False)
    latent_expression_ax = fig.add_subplot(132, frameon=False)
    prediction_ax = fig.add_subplot(133, frameon=False)
    # ax_dict = fig.subplot_mosaic(
    #     [
    #         ["data", "preds"],
    #         ["latent", "preds"],
    #     ],
    # )
    plt.show(block=False)

    for t in range(N_EPOCHS):
        loss, G_means = train(model, model.loss_fn, optimizer)

        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))

            G_means_test, _, _, F_samples_test, = model.forward(
                X_spatial={"expression": x_test},
                view_idx=view_idx_test,
                Ns=Ns_test,
                prediction_mode=True,
                S=10,
            )

            curr_preds = torch.mean(F_samples_test["expression"], dim=0)

            callback_twod(
                model,
                X_train,
                Y_train,
                data_expression_ax=data_expression_ax,
                latent_expression_ax=latent_expression_ax,
                # prediction_ax=ax_dict["preds"],
                X_aligned=G_means,
                # X_test=X_test,
                # Y_test_true=Y_test,
                # Y_pred=curr_preds,
                # X_test_aligned=G_means_test,
            )
            plt.draw()
            plt.pause(1 / 60.0)

            error_gpsa = np.mean(
                np.sum((Y_test - curr_preds.detach().numpy()) ** 2, axis=1)
            )
            print("MSE, GPSA: {}".format(round(error_gpsa, 5)))
            # r2_gpsa = r2_score(Y_test, curr_preds.detach().numpy())
            # print("R2, GPSA: {}".format(round(r2_gpsa, 5)))

            # import ipdb; ipdb.set_trace()
            curr_aligned_coords = G_means["expression"].detach().numpy()
            curr_aligned_coords_test = G_means_test["expression"].detach().numpy()
            # import ipdb; ipdb.set_trace()

            try:
                gpr_gpsa = GaussianProcessRegressor(kernel=RBF() + WhiteKernel())
                gpr_gpsa.fit(X=curr_aligned_coords, y=Y_train)
                preds = gpr_gpsa.predict(curr_aligned_coords_test)
                error_gpsa = np.mean(np.sum((preds - Y_test) ** 2, axis=1))
                print("MSE, GPSA GPR: {}".format(round(error_gpsa, 5)))
            except:
                continue

    errors_gpsa.append(error_gpsa)

    plt.close()

    results_df = pd.DataFrame(
        {
            "Union": errors_union[: repeat_idx + 1],
            "Separate": errors_separate[: repeat_idx + 1],
            "GPSA": errors_gpsa[: repeat_idx + 1],
        }
    )
    results_df_melted = pd.melt(results_df)
    results_df_melted.to_csv("./out/twod_prediction_comparison.csv")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=results_df_melted, x="variable", y="value", color="gray")
    plt.xlabel("")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("../../plots/two_d_prediction_comparison_simulated.png")
    # plt.show()
    plt.close()

import ipdb

ipdb.set_trace()

# import matplotlib

# font = {"size": LATEX_FONTSIZE}
# matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True

# fig = plt.figure(figsize=(10, 10))
# data_expression_ax = fig.add_subplot(211, frameon=False)
# latent_expression_ax = fig.add_subplot(212, frameon=False)
# callback_oned(model, X, Y, data_expression_ax, latent_expression_ax)

# plt.tight_layout()
# plt.savefig("../../plots/one_d_simulation.png")
# plt.show()

# import ipdb

# ipdb.set_trace()
