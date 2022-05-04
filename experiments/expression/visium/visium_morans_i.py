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

sys.path.append("../../..")
sys.path.append("../../../data")
from scipy.sparse import load_npz


## For PASTE
import scanpy as sc
import anndata
from st.load_st_data import load_st_data

import matplotlib

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


sys.path.append("../../../../paste")
from src.paste import PASTE, visualization


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


DATA_DIR = "../../../data/visium/mouse_brain"


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=500)
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


data_slice1 = sc.read_visium(pjoin(DATA_DIR, "sample1"))
data_slice1 = process_data(data_slice1, n_top_genes=6000)

data_slice2 = sc.read_visium(pjoin(DATA_DIR, "sample2"))
data_slice2 = process_data(data_slice2, n_top_genes=6000)

data = data_slice1.concatenate(data_slice2)


sq.gr.spatial_neighbors(data_slice1)
sq.gr.spatial_autocorr(
    data_slice1,
    mode="moran",
)
moran_scores = data_slice1.uns["moranI"]

plt.figure(figsize=(21, 7))
plt.subplot(131)
plt.hist(moran_scores.I.values, 30)
plt.xlim([-0.1, 1])
plt.xlabel(r"$I$")
plt.ylabel("Count")
plt.title("Visium")


DATA_DIR = "../../../data/slideseq/mouse_hippocampus"


spatial_locs_slice1 = pd.read_csv(
    pjoin(DATA_DIR, "Puck_200115_08_spatial_locs.csv"), index_col=0
)
expression_slice1 = load_npz(pjoin(DATA_DIR, "Puck_200115_08_expression.npz"))
gene_names_slice1 = pd.read_csv(
    pjoin(DATA_DIR, "Puck_200115_08_gene_names.csv"), index_col=0
)
barcode_names_slice1 = pd.read_csv(
    pjoin(DATA_DIR, "Puck_200115_08_barcode_names.csv"), index_col=0
)

data_slice1 = anndata.AnnData(
    X=expression_slice1, obs=barcode_names_slice1, var=gene_names_slice1
)
data_slice1.obsm["spatial"] = spatial_locs_slice1.values
data_slice1 = process_data(data_slice1, n_top_genes=6000)


sq.gr.spatial_neighbors(data_slice1)
sq.gr.spatial_autocorr(
    data_slice1,
    mode="moran",
)
moran_scores = data_slice1.uns["moranI"]

plt.subplot(132)
plt.hist(moran_scores.I.values, 30)
plt.xlim([-0.1, 1])
plt.xlabel(r"$I$")
plt.title("Slide-seqV2")


DATA_DIR = "../../../data/st/"
N_LAYERS = 4

data_slice1, data_slice2, data_slice3, data_slice4 = load_st_data(
    layers=np.arange(N_LAYERS) + 1
)
process_data(data_slice1, n_top_genes=3000)

sq.gr.spatial_neighbors(data_slice1)
sq.gr.spatial_autocorr(
    data_slice1,
    mode="moran",
)
moran_scores = data_slice1.uns["moranI"]

plt.subplot(133)
plt.hist(moran_scores.I.values, 30)
plt.xlim([-0.1, 1])
plt.xlabel(r"$I$")
plt.title("ST")

plt.tight_layout()
plt.savefig("./out/morans_i_histograms.png")
plt.show()
plt.close()

import ipdb

ipdb.set_trace()
