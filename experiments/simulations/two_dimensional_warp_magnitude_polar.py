import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from two_dimensional import two_d_gpsa
from scipy.stats import multivariate_normal as mvnpy
import matplotlib

sys.path.append("../..")
from models.gpsa_vi_lmc import VariationalWarpGP

sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data
from warps import apply_polar_warp
from plotting.callbacks import callback_twod
from util import ConvergenceChecker, rbf_kernel
from gp_functions import rbf_covariance

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../paste")
from src.paste import PASTE, visualization

## For PASTE
import scanpy as sc

sys.path.append("../../../paste")
from src.paste import PASTE, visualization


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 50

n_spatial_dims = 2
n_views = 2
n_outputs = 10
m_G = 100
m_X_per_view = 40

PRINT_EVERY = 50
N_LATENT_GPS = {"expression": 3}
n_epochs = 3000


grid_size = 15
xlimits = [0, 10]
ylimits = [0, 10]
x1s = np.linspace(*xlimits, num=grid_size)
x2s = np.linspace(*ylimits, num=grid_size)
X1, X2 = np.meshgrid(x1s, x2s)
X_orig_single = np.vstack([X1.ravel(), X2.ravel()]).T
n_samples_per_view = X_orig_single.shape[0]

n_samples_list = [n_samples_per_view] * n_views
cumulative_sums = np.cumsum(n_samples_list)
cumulative_sums = np.insert(cumulative_sums, 0, 0)
view_idx = np.array(
    [np.arange(cumulative_sums[ii], cumulative_sums[ii + 1]) for ii in range(n_views)]
)
n = np.sum(n_samples_list)

kernel = rbf_covariance
kernel_params_true = [np.log(1.0), np.log(1.0)]
K_XX = kernel(X_orig_single, X_orig_single, kernel_params_true)

nY = N_LATENT_GPS["expression"]

Y_orig = np.vstack(
    [
        mvnpy.rvs(
            mean=np.zeros(X_orig_single.shape[0]),
            cov=K_XX + 0.001 * np.eye(K_XX.shape[0]),
        )
        for _ in range(nY)
    ]
).T


if __name__ == "__main__":
    # coefficient_variance_list = [0.001, 0.005, 0.01]
    coefficient_variance_list = [1e-3, 1e-2, 1e-1]
    n_repeats = 10

    error_mat = np.zeros((n_repeats, len(coefficient_variance_list)))
    error_mat_paste = np.zeros((n_repeats, len(coefficient_variance_list)))

    for ii in range(n_repeats):
        for jj, slope_variance in enumerate(coefficient_variance_list):

            X, Y, n_samples_list, view_idx = apply_polar_warp(
                X_orig_single[:n_samples_per_view],
                Y_orig[:n_samples_per_view],
                n_views=2,
                linear_slope_variance=slope_variance,
                linear_intercept_variance=0.0001,
            )

            ##  PASTE
            slice1 = anndata.AnnData(np.exp(Y[view_idx[0]]))
            slice2 = anndata.AnnData(np.exp(Y[view_idx[1]]))

            slice1.obsm["spatial"] = X[view_idx[0]]
            slice2.obsm["spatial"] = X[view_idx[1]]

            pi12 = PASTE.pairwise_align(slice1, slice2, alpha=0.1)

            slices = [slice1, slice2]
            pis = [pi12]
            new_slices = visualization.stack_slices_pairwise(slices, pis)

            err_paste = np.mean(
                np.sum(
                    (new_slices[0].obsm["spatial"] - new_slices[1].obsm["spatial"])
                    ** 2,
                    axis=1,
                )
            )
            # print(err_paste)

            # plt.subplot(121)
            # plt.scatter(X[view_idx[0]][:, 0], X[view_idx[0]][:, 1])
            # plt.scatter(X[view_idx[1]][:, 0], X[view_idx[1]][:, 1])

            # plt.subplot(122)
            # plt.scatter(new_slices[0].obsm["spatial"][:, 0], new_slices[0].obsm["spatial"][:, 1])
            # plt.scatter(new_slices[1].obsm["spatial"][:, 0], new_slices[1].obsm["spatial"][:, 1])
            # plt.title("PASTE")
            # plt.show()

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
                data_init=True,
                minmax_init=False,
                grid_init=False,
                n_latent_gps=N_LATENT_GPS,
                mean_function="identity_fixed",
                kernel_func_warp=rbf_kernel,
                kernel_func_data=rbf_kernel,
                fixed_warp_kernel_variances=np.ones(n_views) * 1.0,
                fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
            ).to(device)

            view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

            def train(model, loss_fn, optimizer):
                model.train()

                # Forward pass
                G_means, G_samples, F_latent_samples, F_samples = model.forward(
                    {"expression": x}, view_idx=view_idx, Ns=Ns, S=3
                )

                # Compute loss
                loss = loss_fn(data_dict, F_samples)

                # Compute gradients and take optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                return loss.item()

            # Set up figure.
            fig = plt.figure(
                figsize=(14, 7), facecolor="white", constrained_layout=True
            )
            data_expression_ax = fig.add_subplot(122, frameon=False)
            latent_expression_ax = fig.add_subplot(121, frameon=False)
            plt.show(block=False)

            convergence_checker = ConvergenceChecker(span=100)

            loss_trace = []
            error_trace = []
            for t in range(n_epochs):
                loss = train(model, model.loss_fn, optimizer)
                loss_trace.append(loss)
                # print(model.Xtilde)
                # if t >= convergence_checker.span - 1:
                #     rel_change = convergence_checker.relative_change(loss_trace)
                #     is_converged = convergence_checker.converged(loss_trace, tol=1e-5)
                #     if is_converged:
                #         convergence_counter += 1

                #         if convergence_counter == 2:
                #             print("CONVERGED")
                #             break

                #     else:
                #         convergence_counter = 0

                if t % PRINT_EVERY == 0:
                    print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
                    G_means, G_samples, F_latent_samples, F_samples = model.forward(
                        {"expression": x}, view_idx=view_idx, Ns=Ns
                    )

                    callback_twod(
                        model,
                        X,
                        Y,
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
                    err = np.mean(
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
                        ** 2
                    )
                    print("Error: {}".format(err))

                    # if t >= convergence_checker.span - 1:
                    #     print(rel_change)

                G_means, G_samples, F_latent_samples, F_samples = model.forward(
                    {"expression": x}, view_idx=view_idx, Ns=Ns
                )

            aligned_coords = G_means["expression"].detach().numpy().squeeze()
            n_samples_per_view = n_samples_per_view = X.shape[0] // n_views
            view1_aligned_coords = aligned_coords[:n_samples_per_view]
            view2_aligned_coords = aligned_coords[n_samples_per_view:]
            err = np.mean(
                np.sum((view1_aligned_coords - view2_aligned_coords) ** 2, axis=1)
            )

            error_mat[ii, jj] = err
            error_mat_paste[ii, jj] = err_paste

        

        font = {"size": 30}
        matplotlib.rc("font", **font)
        matplotlib.rcParams["text.usetex"] = True
        plt.figure(figsize=(7, 5))

        error_df_gpsa = pd.melt(
            pd.DataFrame(error_mat[: ii + 1, :], columns=coefficient_variance_list)
        )
        error_df_gpsa["method"] = ["GPSA"] * error_df_gpsa.shape[0]
        error_df_paste = pd.melt(
            pd.DataFrame(error_mat_paste[: ii + 1, :], columns=coefficient_variance_list)
        )
        error_df_paste["method"] = ["PASTE"] * error_df_paste.shape[0]

        error_df = pd.concat([error_df_gpsa, error_df_paste], axis=0)
        error_df.to_csv("./out/error_vary_warp_magnitude_polar_warp.csv")

        sns.lineplot(
            data=error_df, x="variable", y="value", hue="method", err_style="bars"
        )
        plt.xlabel("Warp magnitude")
        plt.ylabel("Alignent error")
        plt.title("Polar warp")
        plt.tight_layout()
        plt.savefig("../../plots/two_d_experiments/error_plot_warp_magnitude_polar_warp.png")
        plt.close()

        print("Done!")

        plt.close()
