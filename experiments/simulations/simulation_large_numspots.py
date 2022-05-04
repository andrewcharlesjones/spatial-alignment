import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import pandas as pd

from gpsa import VariationalGPSA
from gpsa import matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod
import sys

sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data

## For PASTE
import scanpy as sc
import anndata
import matplotlib.patches as mpatches

sys.path.append("../../../paste")
from src.paste import PASTE, visualization

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

device = "cuda" if torch.cuda.is_available() else "cpu"

N_SPATIAL_DIMS = 2
N_VIEWS = 2
M_G = 40
M_X_PER_VIEW = 40
N_OUTPUTS = 3
FIXED_VIEW_IDX = 0
N_LATENT_GPS = {"expression": None}

N_EPOCHS = 3_000
PRINT_EVERY = 100

n_latent_gps = {"expression": None}
true_warp_lengthscale = 5.0
true_warp_spatial_variance = 0.1
true_noise_variance = 0.0


n_repeats = 3
grid_size_list = [10, 20, 50]
# grid_size_list = [50]

gpsa_errors = np.zeros((n_repeats, len(grid_size_list)))
paste_errors = np.zeros(n_repeats)

for ii in range(n_repeats):

    for jj, grid_size in enumerate(grid_size_list):

        X, Y, n_samples_list, view_idx = generate_twod_data(
            N_VIEWS,
            N_OUTPUTS,
            grid_size=grid_size,
            n_latent_gps=n_latent_gps["expression"],
            kernel_lengthscale=true_warp_lengthscale,
            kernel_variance=true_warp_spatial_variance,
            noise_variance=true_noise_variance,
        )
        n_samples_per_view = X.shape[0] // N_VIEWS

        ##  PASTE
        # slice1 = anndata.AnnData(np.exp(Y[view_idx[0]]))
        # slice2 = anndata.AnnData(np.exp(Y[view_idx[1]]))

        # slice1.obsm["spatial"] = X[view_idx[0]]
        # slice2.obsm["spatial"] = X[view_idx[1]]

        # pi12 = PASTE.pairwise_align(slice1, slice2, alpha=0.1)

        # slices = [slice1, slice2]
        # pis = [pi12]
        # new_slices = visualization.stack_slices_pairwise(slices, pis)

        # err_paste = np.mean(
        #     np.sum(
        #         (new_slices[0].obsm["spatial"] - new_slices[1].obsm["spatial"]) ** 2, axis=1
        #     )
        # )
        # paste_errors[ii] = err_paste

        ## GPSA
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
            # fixed_warp_kernel_variances=[
            #     fixed_warp_spatial_variance,
            #     fixed_warp_spatial_variance,
            # ],
            # fixed_warp_kernel_lengthscales=[true_warp_lengthscale, true_warp_lengthscale],
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
        # fig = plt.figure(figsize=(14, 7), facecolor="white", constrained_layout=True)
        # data_expression_ax = fig.add_subplot(121, frameon=False)
        # latent_expression_ax = fig.add_subplot(122, frameon=False)
        # plt.show(block=False)

        for t in range(N_EPOCHS):
            loss = train(model, model.loss_fn, optimizer)

            if t % PRINT_EVERY == 0:
                print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
                G_means, _, _, _ = model.forward({"expression": x}, view_idx=view_idx, Ns=Ns)

                # callback_twod(
                #     model,
                #     X,
                #     Y,
                #     data_expression_ax=data_expression_ax,
                #     latent_expression_ax=latent_expression_ax,
                #     X_aligned=G_means,
                #     s=600,
                # )
                # plt.draw()
                # plt.pause(1 / 60.0)

        aligned_coords = G_means["expression"].detach().numpy().squeeze()
        n_samples_per_view = n_samples_per_view = X.shape[0] // N_VIEWS
        view1_aligned_coords = aligned_coords[:n_samples_per_view]
        view2_aligned_coords = aligned_coords[n_samples_per_view:]
        err = np.mean(np.sum((view1_aligned_coords - view2_aligned_coords) ** 2, axis=1))

        gpsa_errors[ii, jj] = err
        plt.close()
        print(err, flush=True)

# import ipdb; ipdb.set_trace()
# results_df = pd.melt(pd.DataFrame({"PASTE": paste_errors, "GPSA": gpsa_errors}))
results_df = pd.melt(pd.DataFrame(gpsa_errors, columns=grid_size_list))
results_df.to_csv("./out/error_experiment_large_numspots.csv")

plt.figure(figsize=(7, 7))
sns.lineplot(data=results_df, x="variable", y="value")
plt.xlabel("Number of spots")
plt.ylabel("Error")
plt.xscale("log")
plt.tight_layout()
plt.savefig("./out/error_experiment_large_numspots.png")
plt.show()
plt.close()



import ipdb
ipdb.set_trace()
        
