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
from sklearn.model_selection import KFold

aligned_coords = pd.read_csv("./out/aligned_coords_visium.csv", index_col=0).values
view_idx = pd.read_csv("./out/view_idx_visium.csv", index_col=0).values
X = pd.read_csv("./out/X_visium.csv", index_col=0).values
Y = pd.read_csv("./out/Y_visium.csv", index_col=0).values
data = sc.read_h5ad("./out/data_visium.h5")

kf = KFold(n_splits=3)
for train_index, test_index in kf.split(aligned_coords):

    ## Joint prediction using aligned coordinates
    X_train, X_test = aligned_coords[train_index], aligned_coords[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    gpr = GaussianProcessRegressor(WhiteKernel() + RBF())
    gpr.fit(X_train, Y_train)
    preds = gpr.predict(X_test)
    curr_r2_aligned = r2_score(Y_test, preds)
    print(curr_r2_aligned)

    ## Joint prediction using aligned coordinates
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    gpr = GaussianProcessRegressor(WhiteKernel() + RBF())
    gpr.fit(X_train, Y_train)
    preds = gpr.predict(X_test)
    curr_r2_unaligned = r2_score(Y_test, preds)
    print(curr_r2_unaligned)
    import ipdb; ipdb.set_trace()




