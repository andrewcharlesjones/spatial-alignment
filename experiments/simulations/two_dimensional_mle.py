import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import sys

sys.path.append("../..")
from models.gpsa_mle import WarpGPMLE

sys.path.append("../../data")
from simulated.generate_twod_data import generate_twod_data
from plotting.callbacks import callback_twod
from util import ConvergenceChecker


device = "cuda" if torch.cuda.is_available() else "cpu"

LATEX_FONTSIZE = 50

n_spatial_dims = 2
n_views = 2
# n_outputs = 10

N_EPOCHS = 3000
PRINT_EVERY = 25
N_LATENT_GPS = 1


def two_d_gpsa(n_outputs, n_epochs, warp_kernel_variance=0.1, plot_intermediate=True):

    X, Y, n_samples_list, view_idx = generate_twod_data(
        n_views,
        n_outputs,
        grid_size=15,
        n_latent_gps=None,
        kernel_lengthscale=10.0,
        kernel_variance=warp_kernel_variance,
        noise_variance=1e-4,
    )
    n_samples_per_view = X.shape[0] // n_views

    ## Fit GP on one view to get initial estimates of data kernel parameters
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.gaussian_process import GaussianProcessRegressor

    kernel = RBF(length_scale=1.0) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X[view_idx[0]], Y[view_idx[0]])
    data_lengthscales_est = gpr.kernel_.k1.theta[0]

    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()

    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }

    model = WarpGPMLE(
        data_dict,
        n_spatial_dims=n_spatial_dims,
        n_latent_gps=None,
        # n_latent_gps=None,
        mean_function="identity_fixed",
        fixed_warp_kernel_variances=np.ones(n_views) * 0.01,
        fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        # fixed_data_kernel_lengthscales=np.exp(gpr.kernel_.k1.theta.astype(np.float32)),
        # fixed_data_kernel_lengthscales=np.exp(data_lengthscales_est),
        # mean_function="identity_initialized",
        fixed_view_idx=0,
    ).to(device)

    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        model.forward({"expression": x}, view_idx=view_idx, Ns=Ns)

        # Compute loss
        loss = loss_fn(
            X_spatial={"expression": x}, view_idx=view_idx, data_dict=data_dict
        )

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # Set up figure.
    fig = plt.figure(figsize=(14, 7), facecolor="white", constrained_layout=True)
    data_expression_ax = fig.add_subplot(122, frameon=False)
    latent_expression_ax = fig.add_subplot(121, frameon=False)
    plt.show(block=False)

    convergence_checker = ConvergenceChecker(span=100)

    loss_trace = []
    error_trace = []

    for t in range(n_epochs):
        loss = train(model, model.loss_fn, optimizer)
        loss_trace.append(loss)
        # print(model.G["expression"][-1])
        # print(torch.exp(model.warp_kernel_variances))
        if t >= convergence_checker.span - 1:
            rel_change = convergence_checker.relative_change(loss_trace)
            is_converged = convergence_checker.converged(loss_trace, tol=1e-4)
            if is_converged:
                convergence_counter += 1

                if convergence_counter == 2:
                    print("CONVERGED")
                    break

            else:
                convergence_counter = 0

        if plot_intermediate and t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
            model.forward({"expression": x}, view_idx=view_idx, Ns=Ns)

            callback_twod(
                model,
                X,
                Y,
                data_expression_ax=data_expression_ax,
                latent_expression_ax=latent_expression_ax,
                X_aligned=model.G,
                is_mle=True,
            )
            plt.draw()
            plt.pause(1 / 60.0)
            err = np.mean(
                (
                    model.G["expression"]
                    .detach()
                    .numpy()
                    .squeeze()[:n_samples_per_view]
                    - model.G["expression"]
                    .detach()
                    .numpy()
                    .squeeze()[n_samples_per_view:]
                )
                ** 2
            )
            print("Error: {}".format(err))

            if t >= convergence_checker.span - 1:
                print(rel_change)

        # G_means, G_samples, F_latent_samples, F_samples = model.forward(
        #     {"expression": x}, view_idx=view_idx, Ns=Ns
        # )

    print("Done!")

    plt.close()

    return X, Y, model.G, model


if __name__ == "__main__":

    n_outputs = 10
    X, Y, G_means, model = two_d_gpsa(n_epochs=N_EPOCHS, n_outputs=n_outputs)

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
    plt.savefig("../../plots/two_d_simulation.png")
    plt.show()

    import ipdb

    ipdb.set_trace()
