import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvnpy
from autograd.scipy.stats import multivariate_normal as mvn
from scipy.stats import multivariate_normal as mvno
import autograd.numpy as np
from autograd import grad, value_and_grad
from autograd.misc.optimizers import adam
from autograd.scipy.special import multigammaln

# from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import pandas as pd

inv = np.linalg.inv
from warp_gp_multigene import rbf_covariance, polar_warp, distance_matrix
from os.path import join as pjoin

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
# matplotlib.rcParams["text.usetex"] = True


def get_coordinates(df):
    """
    Extracts spatial coordinates from ST data with index in 'AxB' type format.

    Return: pandas dataframe of coordinates
    """
    coor = []
    for spot in df.index:
        coordinates = spot.split("x")
        coordinates = [float(i) for i in coordinates]
        coor.append(coordinates)
    return np.array(coor)


if __name__ == "__main__":
    data_dir = "./data/st"
    layer1_path = pjoin(data_dir, "layer1.csv")
    layer2_path = pjoin(data_dir, "layer2.csv")
    layer3_path = pjoin(data_dir, "layer3.csv")
    layer4_path = pjoin(data_dir, "layer4.csv")

    layer1_raw_df = pd.read_csv(layer1_path, index_col=0)
    layer2_raw_df = pd.read_csv(layer2_path, index_col=0)
    layer3_raw_df = pd.read_csv(layer3_path, index_col=0)
    layer4_raw_df = pd.read_csv(layer4_path, index_col=0)

    X_orig_layer1 = get_coordinates(layer1_raw_df)
    X_orig_layer2 = get_coordinates(layer2_raw_df)
    X_orig_layer3 = get_coordinates(layer3_raw_df)
    X_orig_layer4 = get_coordinates(layer4_raw_df)

    ## Select high-variance gene
    gene_vars = layer2_raw_df.var(0).values
    sorted_idx = np.argsort(-gene_vars)
    chosen_idx = sorted_idx[0]
    gene_name = layer2_raw_df.columns.values[chosen_idx]
    # import ipdb; ipdb.set_trace()

    # assert X_orig.shape[0] == Y_orig_unstdized.shape[0]

    plt.figure(figsize=(28, 7))
    plt.subplot(141)
    plt.scatter(
        X_orig_layer1[:, 0], X_orig_layer1[:, 1], c=np.log(layer1_raw_df[gene_name] + 1)
    )
    plt.xlabel("Spatial dim 1")
    plt.ylabel("Spatial dim 2")
    plt.title("ST layer 1")

    plt.subplot(142)
    plt.scatter(
        X_orig_layer2[:, 0], X_orig_layer2[:, 1], c=np.log(layer2_raw_df[gene_name] + 1)
    )
    plt.xlabel("Spatial dim 1")
    plt.ylabel("Spatial dim 2")
    plt.title("ST layer 2")

    plt.subplot(143)
    plt.scatter(
        X_orig_layer3[:, 0], X_orig_layer3[:, 1], c=np.log(layer3_raw_df[gene_name] + 1)
    )
    plt.xlabel("Spatial dim 1")
    plt.ylabel("Spatial dim 2")
    plt.title("ST layer 3")

    plt.subplot(144)
    plt.scatter(
        X_orig_layer4[:, 0], X_orig_layer4[:, 1], c=np.log(layer4_raw_df[gene_name] + 1)
    )
    plt.xlabel("Spatial dim 1")
    plt.ylabel("Spatial dim 2")
    plt.title("ST layer 4")
    plt.tight_layout()
    plt.savefig("./plots/st_data_example.png")
    plt.show()

    import ipdb

    ipdb.set_trace()
