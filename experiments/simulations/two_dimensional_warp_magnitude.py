import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from two_dimensional import two_d_gpsa

# sys.path.append("../..")
# from models.gpsa_vi_lmc import VariationalWarpGP
from gpsa import VariationalGPSA, matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod


sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data
from plotting.callbacks import callback_twod
from util import ConvergenceChecker

## For PASTE
import scanpy as sc

sys.path.append("../../../paste")
from src.paste import PASTE, visualization


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 50

n_spatial_dims = 2
n_views = 2
n_outputs = 10
m_G = 25
m_X_per_view = 25

MAX_EPOCHS = 2000
PRINT_EVERY = -1
N_LATENT_GPS = {"expression": 3}


if __name__ == "__main__":
    kernel_variance_list = [0.01, 0.1, 0.5]
    n_repeats = 3

    error_mat = np.zeros((n_repeats, len(kernel_variance_list)))
    error_mat_paste = np.zeros((n_repeats, len(kernel_variance_list)))

    for ii in range(n_repeats):
        for jj, kernel_variance in enumerate(kernel_variance_list):
            X, Y, G_means, model, err_paste = two_d_gpsa(
                warp_kernel_variance=kernel_variance,
                n_outputs=n_outputs,
                n_epochs=MAX_EPOCHS,
                plot_intermediate=False,
                n_latent_gps=N_LATENT_GPS,
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

            if ii == 0:

                import matplotlib

                font = {"size": LATEX_FONTSIZE}
                matplotlib.rc("font", **font)
                matplotlib.rcParams["text.usetex"] = True

                fig = plt.figure(figsize=(10, 10))
                data_expression_ax = fig.add_subplot(211, frameon=False)
                latent_expression_ax = fig.add_subplot(212, frameon=False)
                callback_twod(
                    model,
                    X,
                    Y,
                    data_expression_ax=data_expression_ax,
                    latent_expression_ax=latent_expression_ax,
                    X_aligned=G_means,
                )

                plt.tight_layout()
                plt.savefig(
                    "../../plots/two_d_experiments/two_d_simulation_kernel_variance={}.png".format(
                        kernel_variance
                    )
                )
                # plt.show()
                plt.close()
                # import ipdb; ipdb.set_trace()

        import matplotlib

        font = {"size": 30}
        matplotlib.rc("font", **font)
        matplotlib.rcParams["text.usetex"] = True
        plt.figure(figsize=(7, 5))

        error_df_gpsa = pd.melt(
            pd.DataFrame(error_mat[: ii + 1, :], columns=kernel_variance_list)
        )
        error_df_gpsa["method"] = ["GPSA"] * error_df_gpsa.shape[0]
        error_df_paste = pd.melt(
            pd.DataFrame(error_mat_paste[: ii + 1, :], columns=kernel_variance_list)
        )
        error_df_paste["method"] = ["PASTE"] * error_df_paste.shape[0]

        error_df = pd.concat([error_df_gpsa, error_df_paste], axis=0)
        error_df.to_csv("./out/error_vary_warp_magnitude.csv")

        sns.lineplot(
            data=error_df, x="variable", y="value", hue="method", err_style="bars"
        )
        plt.xlabel("Warp magnitude")
        plt.ylabel("Alignent error")
        plt.tight_layout()
        plt.savefig("../../plots/two_d_experiments/error_plot_warp_magnitude.png")
        plt.close()

    import ipdb

    ipdb.set_trace()
