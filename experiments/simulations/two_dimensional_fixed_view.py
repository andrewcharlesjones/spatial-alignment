import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from two_dimensional import two_d_gpsa

sys.path.append("../..")
from models.gpsa_vi_lmc import VariationalWarpGP

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
# m_G = 40
# m_X_per_view = 40

MAX_EPOCHS = 2000
PRINT_EVERY = 25
N_LATENT_GPS = {"expression": 5}


if __name__ == "__main__":
    n_repeats = 3

    for ii in range(n_repeats):
        X, Y, G_means, model, err_paste = two_d_gpsa(
            n_outputs=n_outputs,
            n_epochs=MAX_EPOCHS,
            plot_intermediate=True,
            warp_kernel_variance=0.5,
            n_latent_gps=N_LATENT_GPS,
            fixed_view_idx=0,
        )

    import ipdb

    ipdb.set_trace()
