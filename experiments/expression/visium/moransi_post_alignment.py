import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import sys
from os.path import join as pjoin
import scanpy as sc
import squidpy as sq
import anndata
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.patches as patches

sys.path.append("../../..")
sys.path.append("../../../data")
# from plotting.callbacks import callback_oned, callback_twod
from gpsa.plotting import callback_oned, callback_twod

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern
from sklearn.model_selection import KFold

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["xtick.labelsize"] = 20
matplotlib.rcParams["ytick.labelsize"] = 20

aligned_coords = pd.read_csv("./out/aligned_coords_visium.csv", index_col=0).values
view_idx = pd.read_csv("./out/view_idx_visium.csv", index_col=0).values
# X = pd.read_csv("./out/X_visium.csv", index_col=0).values
# Y = pd.read_csv("./out/Y_visium.csv", index_col=0).values
# data = sc.read_h5ad("./out/data_visium.h5")

# data_aligned = data.copy()
# data_aligned.obsm["spatial"] = aligned_coords


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


DATA_DIR = "../../../data/visium/mouse_brain"

n_spatial_dims = 2
n_views = 2


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


if __name__ == "__main__":
    data_slice1 = sc.read_visium(pjoin(DATA_DIR, "sample1"))
    data_slice1 = process_data(data_slice1, n_top_genes=6000)

    data_slice2 = sc.read_visium(pjoin(DATA_DIR, "sample2"))
    data_slice2 = process_data(data_slice2, n_top_genes=6000)

    data = data_slice1.concatenate(data_slice2)

    data_slice1 = data[data.obs.batch == "0"]
    data_slice2 = data[data.obs.batch == "1"]

    data_aligned = anndata.AnnData(np.array(data.X.todense()).copy())
    data_aligned.obsm["spatial"] = aligned_coords
    data_aligned.var_names = data.var_names

    ## Compute Moran's I

    sq.gr.spatial_neighbors(data)
    sq.gr.spatial_autocorr(
        data,
        mode="moran",
    )
    moran_scores_union = data.uns["moranI"]

    sq.gr.spatial_neighbors(data_aligned)
    sq.gr.spatial_autocorr(
        data_aligned,
        mode="moran",
    )
    moran_scores_gpsa = data_aligned.uns["moranI"]

    # moran_scores_gpsa = moran_scores_gpsa[moran_scores_gpsa.pval_norm > 0]
    # moran_scores_union = moran_scores_union[moran_scores_union.pval_norm > 0]

    moran_scores = pd.merge(
        moran_scores_union,
        moran_scores_gpsa,
        left_index=True,
        right_index=True,
        suffixes=["_union", "_gpsa"],
    )

    plt.figure(figsize=(10, 7))

    # plt.subplot(121)
    # plt.scatter(moran_scores.pval_norm_fdr_bh_union, moran_scores.pval_norm_fdr_bh_gpsa)
    # plt.scatter(moran_scores.I_union, moran_scores.I_gpsa, c=moran_scores.pval_norm_fdr_bh_gpsa < moran_scores.pval_norm_fdr_bh_union)

    # new_and_significant = np.logical_and(
    #     (moran_scores.pval_norm_fdr_bh_gpsa < 0.01).values,
    #     (
    #         moran_scores.pval_norm_fdr_bh_gpsa < moran_scores.pval_norm_fdr_bh_union
    #     ).values,
    # )
    new_and_significant = np.logical_and(
        (moran_scores.pval_norm_fdr_bh_gpsa < 0.01).values,
        (moran_scores.pval_norm_fdr_bh_union > 0.01).values,
    )
    old_and_significant = np.logical_and(
        (moran_scores.pval_norm_fdr_bh_gpsa < 0.01).values,
        (moran_scores.pval_norm_fdr_bh_union < 0.01).values,
    )
    moran_scores["new_and_significant"] = new_and_significant

    # new_and_significant_strs = []
    # for ii in range(len(moran_scores)):
    #     if new_and_significant[ii]:
    #         new_and_significant_strs.append("GPSA hit\n" + r"($p < 0.01$)")
    #     elif old_and_significant[ii]:
    #         new_and_significant_strs.append("Pre-alignment hit\n" + r"($p < 0.01$)")
    #     else:
    #         new_and_significant_strs.append("")
    new_and_significant_strs = np.array(
        [
            "GPSA-specific hit\n" + r"($p < 0.01$)" if x else ""
            for x in new_and_significant
        ]
    )
    # new_and_significant_strs[np.where(old_and_significant)[0]] = "Prealignment hit\n" + r"($p < 0.01$)"
    moran_scores["new_and_significant_str"] = new_and_significant_strs
    # moran_scores[moran_scores["new_and_significant"] == True]["new_and_significant"] = r"GPSA hit ($p < 0.01$)"
    # moran_scores[moran_scores["new_and_significant"] == False]["new_and_significant"] = ""
    g = sns.scatterplot(
        data=moran_scores,
        x="I_union",
        y="I_gpsa",
        hue="new_and_significant_str",
        edgecolor=None,
    )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=20)
    g.legend_.set_title(None)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

    plt.xlabel(r"Moran's $I$, Union")
    plt.ylabel(r"Moran's $I$, GPSA")
    # plt.colorbar()
    plt.tight_layout()
    plt.savefig("./out/moransi_post_alignment.png")
    plt.show()

    print(
        (moran_scores[["pval_norm_fdr_bh_union", "pval_norm_fdr_bh_gpsa"]] < 1e-4).sum(
            0
        )
    )
    # import ipdb; ipdb.set_trace()

    # plt.subplot(122)
    # plt.scatter(-np.log10(moran_scores.pval_norm_fdr_bh_union + 1e-8), -np.log10(moran_scores.pval_norm_fdr_bh_gpsa + 1e-8))

    # plt.show()

    # plt.scatter(moran_scores.I_union, -np.log10(moran_scores.pval_norm_fdr_bh_union + 1e-8))
    # plt.scatter(moran_scores.I_gpsa, -np.log10(moran_scores.pval_norm_fdr_bh_gpsa + 1e-8))
    # plt.show()

    import ipdb

    ipdb.set_trace()
